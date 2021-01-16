# 把预训练好的模型加载进来，并修改输出层
import json
import logging
from termcolor import colored
import modeling
import args
import tensorflow as tf
import os


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def optimize_graph(logger=None, verbose=False):
    if not logger:
        logger = set_logger(colored('BERT_VEC', 'yellow'), verbose)
    try:
        # we don't need GPU for optimizing the graph
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
        tf.gfile.MakeDirs(args.output_dir) # 创建一个目录，该目录用于存放待输出的文件，

        config_fp = args.config_name
        logger.info('model config: %s' % config_fp)

        # 加载bert配置文件
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))
            #  定义bert的配置信息路径，根据路径读取配置信息转化为bert_config对象

        logger.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
            # 定义三个占位符，分别表示的是对应文本的index，mask与type_index，
            # 其中index表示的是在词典中的index，mask表示的是该位置是否有内容，
            # 举个例子，例如序列的最大长度是20，有效的字符只有10个字，加上[CLS]与[SEP]两个占位符，
            # 那有8个字符是空的，该8个位置设置为0其他位置设置为1，type_index表示的是是否是第一个句子，
            # 是第一个句子则设置为1，因为该项目只有一个句子，所以均为1
        input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_type_ids')

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

        with jit_scope():
            # 根据上面定义的三个占位符，定义好输入的张量，实例化一个model对象，该对象就是预训练好的bert模型，然后从check_point文件中初始化权重
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            # 获取所有要训练的变量
            tvars = tf.trainable_variables()

            init_checkpoint = args.ckpt_name
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            # 接下来判断一下args.index_layeres 参数的长度，如果长度为1，则只取改层的输出，否则遍历需要取的层，把所有层的weight取出来并拼接成一个768*层数的张量
            # 共享卷积核
            with tf.variable_scope("pooling"):
                # 如果只有一层，就只取对应那一层的weight
                if len(args.layer_indexes) == 1:
                    encoder_layer = model.all_encoder_layers[args.layer_indexes[0]]
                else:
                    # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
                    all_layers = [model.all_encoder_layers[l] for l in args.layer_indexes]
                    encoder_layer = tf.concat(all_layers, -1)

# ------------------------------------------------------------------------------------------------------------------------------------------
# 接下来是句向量生成的核心代码，这里定义了两个方法，一个mul_mask 和一个masked_reduce_mean，我们先看masked_reduce_mean(encoder_layer, input_mask)
# 这里调用方法时传入的是encoder_layer即输出值，与input_mask即是否有有效文本，masked_reduce_mean方法中又调用了mul_mask方法，即先把input_mask进行了一个维度扩展，
# 然后与encoder_layer相乘，为什么要维度扩展呢，我们看下两个值的维度，我们还是假设序列的最大长度是20，那么encoder_layer的维度为[20,768]，为了把无效的位置的内容置为0，
# input_mask的维度为[20]，扩充之后变成了[20，1]，两个值相乘，便把input_mask为0的位置的encoder_layer的值改为了0， 然后把相乘得到的值在axis=1的位置进行相加
# 最后除以input_mask在axis=1的维度的和，然后把得到的结果添加一个别名final_encodes
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            input_mask = tf.cast(input_mask, tf.float32)
            # 以下代码是句向量的生成方法，可以理解为做了一个卷积的操作，但是没有把结果相加, 卷积核是input_mask
            pooled = masked_reduce_mean(encoder_layer, input_mask)
            pooled = tf.identity(pooled, 'final_encodes')

            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()
# ------------------------------------------------------------------------------------------------------------------------------------------

#   最后把得到的句向量重新添加进graph中，并返回graph的路径
        # allow_soft_placement:自动选择运行设备
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...')
            tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        # tmp_file = tempfile.NamedTemporaryFile('w', delete=False, dir=args.output_dir).name
        tmp_file = args.graph_file
        logger.info('write graph to a tmp file: %s' % tmp_file)
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return tmp_file
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)
