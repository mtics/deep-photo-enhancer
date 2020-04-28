import sys

exp = 999

sys.path.append('LPGAN_exp_G3_%d/' % exp)

from DATA  import *
from MODEL import *

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS['num_gpu']
netG_act_o = dict(size=1, index=0)
netG = NetInfo('netG')
test_df = DataFlow()
with tf.name_scope(netG.name):
    with tf.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_test_output1 = model(netG, test_df.input1, False, netG_act_o, is_first=True)

new_netG = NetInfo('netG-%d' % FLAGS['num_exp'])
with tf.name_scope(new_netG.name):
    with tf.variable_scope(new_netG.variable_scope_name) as scope_full:
        with tf.variable_scope(new_netG.variable_scope_name + 'A') as scopeA:
            new_netG_test_output1 = model(new_netG, test_df.input1, False, netG_act_o, is_first=True)

assert len(netG.weights) == len(netG.parameter_names), 'len(weights) != len(parameters)'
saver = tf.train.Saver(var_list=netG.weights, max_to_keep=None)

sess_config = tf.ConfigProto(log_device_placement=False)
sess_config.gpu_options.allow_growth = True

sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
saver.restore(sess, FLAGS['load_model_path'])

new_weights = []
for n_v, o_v in zip(new_netG.weights, netG.weights):
    new_weights.append(tf.assign(n_v, o_v))

sess.run(new_weights)
assert len(new_netG.weights) == len(new_netG.parameter_names), 'len(weights) != len(parameters)'
saver = tf.train.Saver(var_list=new_netG.weights, max_to_keep=None)
saver.save(sess, FLAGS['load_model_path_new'])

