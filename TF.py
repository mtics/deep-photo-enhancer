import os, sys
import tensorflow as tf

from DATA import *
from MODEL import *
from FUNCTION import *
from EVALUATION import *

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS['num_gpu']
sys.stdout = Tee(sys.stdout, open(FLAGS['txt_log'], 'a+'))

LOG_STAMP_TRAIN = np.int32(np.linspace(0, FLAGS['data_train_batch_count']-1, num=FLAGS['process_train_log_interval_epoch']+1))
LOG_STAMP_TEST  = np.int32(np.linspace(0, FLAGS['data_train_batch_count']-1, num=FLAGS['process_test_log_interval_epoch'] +1))

def netG_concat_value(tensor, v):
    v_t = tf.constant(v, dtype=tf.float32, shape=tensor.get_shape().as_list()[:3] + [1])
    tensor = tf.concat(3, [tensor, v_t])
    return tensor

netG_act_o = dict(size=1, index=0)

with tf.name_scope(netG.name):
    with tf.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_train_output1 = model(netG, train_df.input1, True, netG_act_o, is_first=True)
            scopeA.reuse_variables()
            netG_test_output1  = model(netG, test_df.input1, False, netG_act_o)

save_net(netG.parameter_names, FLAGS['netG_mat'])

assert len(netG.weights) == len(netG.parameter_names), 'len(weights) != len(parameters)'
saver = tf.train.Saver(var_list=netG.weights, max_to_keep=None)

netG_r_loss = regularization_cost(netG)
netG_w_regularization_loss = netG_r_loss * netG.REGULARIZATION_WEIGHT

with tf.name_scope("Loss"):
    netG_train_output1_crop = [tf_crop_rect(netG_train_output1, train_df.mat1, i) for i in range(FLAGS['data_train_batch_size'])]
    netG_train_input2_crop  = [tf_crop_rect(train_df.input2,    train_df.mat2, i) for i in range(FLAGS['data_train_batch_size'])]
    netG_test_output1_crop  =  tf_crop_rect(netG_test_output1, test_df.mat1, 0)
    netG_test_input2_crop   =  tf_crop_rect(test_df.input2,    test_df.mat2, 0)

    if FLAGS['loss_source_data_term'] == 'l2':
        train_data_term_1 = tf.pack([-img_L2_loss(a, b, FLAGS['loss_data_term_use_local_weight']) for a, b in zip(netG_train_output1_crop, netG_train_input2_crop)])
        test_data_term_1  = -img_L2_loss(netG_test_output1_crop, netG_test_input2_crop, FLAGS['loss_data_term_use_local_weight'])
    elif FLAGS['loss_source_data_term'] == 'l1':
        train_data_term_1 = tf.pack([img_L1_loss(a, b) for a, b in zip(netG_train_output1_crop, netG_train_input2_crop)])
        test_data_term_1  = -img_L1_loss(netG_test_output1_crop, netG_test_input2_crop)
    elif FLAGS['loss_source_data_term'] == 'PR':
        assert False, 'not yet'
        train_data_term_1 =  tf.pack([tf_photorealism_loss(netG_train_output1, train_df.mat1, i, FLAGS['loss_photorealism_is_our']) for i in range(FLAGS['data_train_batch_size'])])
        train_data_term_1 = -tf.reduce_mean(train_data_term_1) * FLAGS['loss_source_data_term_weight']
        test_data_term_1  =  tf.pack([tf_photorealism_loss(netG_test_output1,  test_df.mat1,  0, FLAGS['loss_photorealism_is_our'])])
        test_data_term_1  = -tf.reduce_mean(test_data_term_1)  * FLAGS['loss_source_data_term_weight']
    elif FLAGS['loss_source_data_term'] == 'PSNR':
        train_data_term_1 = -tf_log10(tf.pack([img_L2_loss(a, b, FLAGS['loss_data_term_use_local_weight']) for a, b in zip(netG_train_output1_crop, netG_train_input2_crop)]))
        test_data_term_1  = -tf_log10(img_L2_loss(netG_test_output1_crop, netG_test_input2_crop, FLAGS['loss_data_term_use_local_weight']))
    else:
        assert False, 'data term error = %s' % FLAGS['loss_source_data_term']

    netG_batch_list_train_loss = train_data_term_1
    netG_train_loss = tf.reduce_mean(train_data_term_1)
    netG_test_loss = test_data_term_1
    netG_loss = netG_train_loss

netG_total_loss = -netG_loss + netG_w_regularization_loss

with tf.name_scope("netG_SGD"):
    netG_optimizer = netG.OPTIMIZER
    netG_gvs = netG_optimizer.compute_gradients(netG_total_loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=netG.variable_scope_name))
    netG_gbc = [grad for grad, var in netG_gvs]
    netG_capped_gvs = [(tf.clip_by_value(grad, -netG.GLOBAL_GRADIENT_CLIPPING, netG.GLOBAL_GRADIENT_CLIPPING), var) for grad, var in netG_gvs]
    netG_gac = [grad for grad, var in netG_capped_gvs]
    netG_opt = netG_optimizer.apply_gradients(netG_capped_gvs)

    netG_gbc = tf.reduce_mean(tf.pack([tf.reduce_mean(tf.abs(v)) for v in netG_gbc]))
    netG_gac = tf.reduce_mean(tf.pack([tf.reduce_mean(tf.abs(v)) for v in netG_gac]))

netG_train_output1_crop_rounds = [tf.cast(tf.round(tf.clip_by_value(a, 0, 1) * train_df.input2_src.dtype.max), tf.as_dtype(FLAGS['data_label_dtype'])) for a in netG_train_output1_crop]
netG_train_input2_crop_rounds  = [tf.cast(tf.round(tf.clip_by_value(a, 0, 1) * train_df.input2_src.dtype.max), tf.as_dtype(FLAGS['data_label_dtype'])) for a in netG_train_input2_crop]
netG_train_psnr1 = tf.reduce_mean([tf_psnr(a, b, 0) for a, b in zip(netG_train_output1_crop_rounds, netG_train_input2_crop_rounds)])

netG_test_output1_crop_round = tf.cast(tf.round(tf.clip_by_value(netG_test_output1_crop, 0, 1) * test_df.input2_src.dtype.max), tf.as_dtype(FLAGS['data_label_dtype']))
netG_test_input2_crop_round  = tf.cast(tf.round(tf.clip_by_value(netG_test_input2_crop,  0, 1) * test_df.input2_src.dtype.max), tf.as_dtype(FLAGS['data_label_dtype']))
netG_test_psnr1 = tf_psnr(netG_test_output1_crop_round, netG_test_input2_crop_round, 0)

netG_train_summary = [netG_train_loss, netG_train_psnr1, netG_r_loss, netG_gbc, netG_gac]
netG_test_summary  = [ netG_test_loss, netG_test_psnr1]
netG_train_summary_names = ["G_tr_l", "G_tr_psnr1", "G_r", "G_gbc", "G_gac"]
netG_test_summary_names  = ["G_te_l", "G_psnr1"]
train_summary_names = netG_train_summary_names
test_summary_names  =  netG_test_summary_names

assert(len(netG_train_summary) == len(netG_train_summary_names))
assert(len(netG_test_summary) == len(netG_test_summary_names))
train_summary_placeholders = []
test_summary_placeholders = []
train_summary_list = []
test_summary_list = []
for name in train_summary_names:
    train_summary_placeholders.append(tf.placeholder(tf.as_dtype(FLAGS['data_compute_dtype'])))
    train_summary_list.append(tf.summary.scalar(name, train_summary_placeholders[-1]))
for name in test_summary_names:
    test_summary_placeholders.append(tf.placeholder(tf.as_dtype(FLAGS['data_compute_dtype'])))
    test_summary_list.append(tf.summary.scalar(name, test_summary_placeholders[-1]))

def update_cache_dict(csr_ind, csr_val, csr_ind_r, csr_val_r, csr_ind_g, csr_val_g, csr_ind_b, csr_val_b, csr_names):
    for i, names in enumerate(csr_names):
        for name in names:
            if name[-2:] == '_r':
                csr_ind_r[i] = csr_dict[name][:, :2] - 1
                csr_val_r[i] = csr_dict[name][:, -1]
            elif name[-2:] == '_g':
                csr_ind_g[i] = csr_dict[name][:, :2] - 1
                csr_val_g[i] = csr_dict[name][:, -1]
            elif name[-2:] == '_b':
                csr_ind_b[i] = csr_dict[name][:, :2] - 1
                csr_val_b[i] = csr_dict[name][:, -1]
            else:
                csr_ind[i] = csr_dict[name][:, :2] - 1
                csr_val[i] = csr_dict[name][:, -1]
def do_testing(now_epoch, data_loader, best_value_history, indices_1, indices_2, list_train_loss, \
    summary_writer, train_summary_datas, is_training):
    timer = Timer()
    timer.start()
    test_avg = [0] * len(netG_test_summary_names)
    test_count = FLAGS['data_test_image_count'] if FLAGS['process_write_test_img_count'] == 0 or is_training == False else FLAGS['process_write_test_img_count']

    test_loss_list = np.zeros((len(netG_test_summary_names), test_count))
    for i in range(test_count):
        timer2 = Timer()
        timer2.start()
        label_img = data_loader.get_next_test_label()
        input_img, data = data_loader.get_next_test_input_batch()
        update_cache_dict(data['csr_ind1'], data['csr_val1'], data['csr_ind_r1'], data['csr_val_r1'], data['csr_ind_g1'], data['csr_val_g1'], data['csr_ind_b1'], data['csr_val_b1'], data['csr_names1'])
        update_cache_dict(data['csr_ind2'], data['csr_val2'], data['csr_ind_r2'], data['csr_val_r2'], data['csr_ind_g2'], data['csr_val_g2'], data['csr_ind_b2'], data['csr_val_b2'], data['csr_names2'])

        dict_d = [\
            input_img, label_img] + \
            data['rect1'] + data['rot1'] + \
            data['rect2'] + data['rot2'] + \
            data['csr_ind1']   + data['csr_val1'] + \
            data['csr_ind_r1'] + data['csr_val_r1'] + \
            data['csr_ind_g1'] + data['csr_val_g1'] + \
            data['csr_ind_b1'] + data['csr_val_b1'] + data['csr_sha1'] + \
            data['csr_ind2']   + data['csr_val2'] + \
            data['csr_ind_r2'] + data['csr_val_r2'] + \
            data['csr_ind_g2'] + data['csr_val_g2'] + \
            data['csr_ind_b2'] + data['csr_val_b2'] + data['csr_sha2']
        dict_t = [\
            test_df.input1_src, test_df.input2_src] + \
            test_df.mat1.rect + test_df.mat1.rot + \
            test_df.mat2.rect + test_df.mat2.rot + \
            test_df.mat1.csr_ind   + test_df.mat1.csr_val + \
            test_df.mat1.csr_ind_r + test_df.mat1.csr_val_r + \
            test_df.mat1.csr_ind_g + test_df.mat1.csr_val_g + \
            test_df.mat1.csr_ind_b + test_df.mat1.csr_val_b + test_df.mat1.csr_sha + \
            test_df.mat2.csr_ind   + test_df.mat2.csr_val + \
            test_df.mat2.csr_ind_r + test_df.mat2.csr_val_r + \
            test_df.mat2.csr_ind_g + test_df.mat2.csr_val_g + \
            test_df.mat2.csr_ind_b + test_df.mat2.csr_val_b + test_df.mat2.csr_sha

        test_s, enhance_test_img = sess.run([netG_test_summary, netG_test_output1_crop], feed_dict={t:d for t, d in zip(dict_t, dict_d)})
        enhance_test_img = safe_casting(enhance_test_img * tf.as_dtype(FLAGS['data_input_dtype']).max, FLAGS['data_input_dtype'])
        if is_training:
            cv2.imwrite(FLAGS['folder_test_img'] + test_image_name_list[i] + '-' + now_epoch + FLAGS['data_input_ext'], enhance_test_img)
        else:
            cv2.imwrite(FLAGS['folder_test_img'] + test_image_name_list[i] + FLAGS['data_input_ext'], enhance_test_img)
        
        if FLAGS['mode_use_debug']:
            print("i = %d, %s" % (i, str(timer2.end())))

        for j in range(len(netG_test_summary_names)):
            test_loss_list[j][i] = test_s[j]
        test_avg = [test_avg[j] + v for j, v in enumerate(test_s)]

    test_avg = [v / test_count for v in test_avg]
    best_value_history = [max(v, test_avg[i]) for i, v in enumerate(best_value_history)]

    if is_training:
        step = round(float(now_epoch)*FLAGS['process_test_log_interval_epoch'])
        if step >= FLAGS['process_test_drop_summary_step']:
            for i, v in enumerate(test_avg):
                summary = sess.run(test_summary_list[i], feed_dict={test_summary_placeholders[i]: v})
                summary_writer.add_summary(summary, step)

    # weights_data = sess.run(main_net.weights)
    # save_weights(weights_data, main_net.parameter_names, FLAGS['weight_folder'], now_epoch)

    if is_training:
        write_list_to_file(FLAGS['folder_test_netG_loss']  + now_epoch + ".txt", test_loss_list[ 0], test_image_name_list[:test_count])
        write_list_to_file(FLAGS['folder_test_netG_psnr1'] + now_epoch + ".txt", test_loss_list[-1], test_image_name_list[:test_count])
        write_list_to_file(FLAGS['folder_train_netG_loss'] + now_epoch + ".txt", list_train_loss, train_image_name_list_input)
        write_list_to_file(FLAGS['folder_train_ind_input'] + now_epoch + ".txt", indices_1, train_image_name_list_input)
        write_list_to_file(FLAGS['folder_train_ind_label'] + now_epoch + ".txt", indices_2, train_image_name_list_label)
        save_model(saver, sess, FLAGS['folder_model'], now_epoch)

        for train_c_summary, step in train_summary_datas:
            summary_writer.add_summary(train_c_summary, step)

    info = current_time() + ", epoch = " + now_epoch
    for i, v in enumerate(test_avg):
        info = info + ", %s = %s" % (netG_test_summary_names[i], FLAGS['format_log_value'].format(v).replace(' ', '*'))
    info = info + ", (max:"
    for i, v in enumerate(best_value_history):
        info = info + "%s" % FLAGS['format_log_value'].format(v).replace(' ', '*')
        info = info + (")" if i == len(best_value_history) - 1 else ", ")
    info = info + ", tt = " + str(timer.end()).split(".")[0][2:]
    print(info)
    return best_value_history, []

def do_training_log(sess, train_summary_datas, now_epoch, train_avg, data_run_time, sess_run_time):
    step = round(float(now_epoch)*FLAGS['process_train_log_interval_epoch'])
    if step >= FLAGS['process_train_drop_summary_step']:
        for i, v in enumerate(train_avg):
            summary = sess.run(train_summary_list[i], feed_dict={train_summary_placeholders[i]: v})
            train_summary_datas.append((summary, step))

    info = current_time() + ", epoch = " + now_epoch
    for i, v in enumerate(train_avg):
        info = info + ", %s = %s" % (train_summary_names[i], FLAGS['format_log_value'].format(v))
    info = info + ", dt = " + str(data_run_time).split(".")[0][2:] + ", st = " + str(sess_run_time).split(".")[0][2:]
    print(info)

    return timer.zero_time(), timer.zero_time(), train_summary_datas

sess_config = tf.ConfigProto(log_device_placement=False)                                                                                                                                               
sess_config.gpu_options.allow_growth = (FLAGS['sys_use_all_gpu_memory'] == False)                                                                                                                 

if __name__ == '__main__':
    with tf.Session(config=sess_config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(FLAGS['folder_log'], sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        timer = Timer()
        data_run_time = timer.zero_time()
        sess_run_time = timer.zero_time()
        data_loader = DataLoader()
        prepare_csr_dict(csr_dict, True)

        print(current_time() + ", [netG] architecture_log :")
        for s in netG.architecture_log:
            print(current_time() + ", %s" % s)

        start_epoch = int(FLAGS['process_epoch'])
        remainder = FLAGS['process_epoch'] - start_epoch
        start_iter = 0
        if remainder != 0:
            stamp_index_TRAIN = int(round(remainder * FLAGS['process_train_log_interval_epoch'])) + 1
            stamp_index_TEST  = int(round(remainder * FLAGS['process_test_log_interval_epoch']))  + 1
            start_iter        = LOG_STAMP_TEST[stamp_index_TEST - 1]                      + 1

        list_train_loss = [-1e8] * FLAGS['data_train_sample_count']
        indices_1 = list(range(FLAGS['data_train_sample_count']))
        random.shuffle(indices_1)
        indices_2 = list(range(FLAGS['data_train_sample_count']))
        random.shuffle(indices_2)

        if FLAGS['load_model_need']:
            print(current_time() + ", Train loss data loading...: %s" % FLAGS['load_train_loss_path'])
            list_train_loss = read_file_to_list(FLAGS['load_train_loss_path'], float)
            print(current_time() + ", Train loss data restored from file: %s" % FLAGS['load_train_loss_path'])

            print(current_time() + ", Train indices input data loading...: %s" % FLAGS['load_train_indices_input_path'])
            indices_1 = read_file_to_list(FLAGS['load_train_indices_input_path'], int)
            print(current_time() + ", Train indices input data restored from file: %s" % FLAGS['load_train_indices_input_path'])

            print(current_time() + ", Train indices label data loading...: %s" % FLAGS['load_train_indices_label_path'])
            indices_2 = read_file_to_list(FLAGS['load_train_indices_label_path'], int)
            print(current_time() + ", Train indices label data restored from file: %s" % FLAGS['load_train_indices_label_path'])

            print(current_time() + ", Model loading...: %s" % FLAGS['load_model_path'])
            saver.restore(sess, FLAGS['load_model_path'])
            print(current_time() + ", Model restored from file: %s" % FLAGS['load_model_path'])

        if FLAGS['process_train_data_loader_count'] > 0:
            data_loader.enqueue_train_indices_by_data(indices_1, indices_2, start_iter)

        flag_keys = FLAGS.keys()
        flag_keys = sorted(flag_keys)
        for key in flag_keys:
            print(current_time() + ', [%s] : %s' % (key, str(FLAGS[key])))

        best_value_history = [-1e8] * len(netG_test_summary_names)
        train_summary_datas = []
        if FLAGS['process_train_data_loader_count'] <= 0 or (FLAGS['process_run_first_testing_epoch'] and FLAGS['process_epoch'] == 0):
           best_value_history, train_summary_datas = do_testing(FLAGS['format_log_step'] % 0, data_loader, best_value_history, indices_1, indices_2, list_train_loss, \
            summary_writer, train_summary_datas, FLAGS['process_train_data_loader_count'] > 0)

        if FLAGS['process_train_data_loader_count'] > 0:
            total_iter = start_epoch * FLAGS['data_train_batch_count'] 
            netG_base_lr = FLAGS['netG_base_learning_rate']
            for epoch in range(start_epoch, FLAGS['process_max_epoch']):
                if epoch != start_epoch or remainder == 0:
                    start_iter = 0
                    stamp_index_TRAIN = 0 if epoch == 0 else 1
                    stamp_index_TEST = 1

                netG_train_avg = [0] * len(netG_train_summary_names)
                netG_train_c_count = 0

                for iter in range(start_iter, FLAGS['data_train_batch_count']):
                    timer.start()
                    data = data_loader.get_next_train_batch()
                    update_cache_dict(data['csr_ind1'], data['csr_val1'], data['csr_ind_r1'], data['csr_val_r1'], data['csr_ind_g1'], data['csr_val_g1'], data['csr_ind_b1'], data['csr_val_b1'], data['csr_names1'])
                    update_cache_dict(data['csr_ind2'], data['csr_val2'], data['csr_ind_r2'], data['csr_val_r2'], data['csr_ind_g2'], data['csr_val_g2'], data['csr_ind_b2'], data['csr_val_b2'], data['csr_names2'])
                    data_run_time = data_run_time + timer.end()

                    dict_d = [\
                        data['input'], data['label'], netG_base_lr] + \
                        data['rect1'] + data['rot1'] + \
                        data['rect2'] + data['rot2'] + \
                        data['csr_ind1']   + data['csr_val1'] + \
                        data['csr_ind_r1'] + data['csr_val_r1'] + \
                        data['csr_ind_g1'] + data['csr_val_g1'] + \
                        data['csr_ind_b1'] + data['csr_val_b1'] + data['csr_sha1'] + \
                        data['csr_ind2']   + data['csr_val2'] + \
                        data['csr_ind_r2'] + data['csr_val_r2'] + \
                        data['csr_ind_g2'] + data['csr_val_g2'] + \
                        data['csr_ind_b2'] + data['csr_val_b2'] + data['csr_sha2']
                    dict_t = [\
                        train_df.input1_src, train_df.input2_src, netG.base_lr] + \
                        train_df.mat1.rect + train_df.mat1.rot + \
                        train_df.mat2.rect + train_df.mat2.rot + \
                        train_df.mat1.csr_ind   + train_df.mat1.csr_val + \
                        train_df.mat1.csr_ind_r + train_df.mat1.csr_val_r + \
                        train_df.mat1.csr_ind_g + train_df.mat1.csr_val_g + \
                        train_df.mat1.csr_ind_b + train_df.mat1.csr_val_b + train_df.mat1.csr_sha + \
                        train_df.mat2.csr_ind   + train_df.mat2.csr_val + \
                        train_df.mat2.csr_ind_r + train_df.mat2.csr_val_r + \
                        train_df.mat2.csr_ind_g + train_df.mat2.csr_val_g + \
                        train_df.mat2.csr_ind_b + train_df.mat2.csr_val_b + train_df.mat2.csr_sha

                    _, netG_train_s, batch_list_train_c = sess.run([\
                        netG_opt, netG_train_summary, netG_batch_list_train_loss], \
                            feed_dict={t:d for t, d in zip(dict_t, dict_d)})
                    netG_train_avg = [netG_train_avg[i] + v for i, v in enumerate(netG_train_s)]
                    netG_train_c_count += 1
                    sess_run_time = sess_run_time + timer.end()

                    for i in range(len(batch_list_train_c)):
                        list_train_loss[indices_1[iter*FLAGS['data_train_batch_size']+i]] = batch_list_train_c[i]

                    now_epoch = FLAGS['format_log_step'] % (epoch + (iter + 1.0) / FLAGS['data_train_batch_count'])

                    if iter == LOG_STAMP_TRAIN[stamp_index_TRAIN]:
                        netG_train_avg = [netG_train_avg[i] / (netG_train_c_count if netG_train_c_count != 0 else 1) for i, v in enumerate(netG_train_avg)]
                        data_run_time, sess_run_time, train_summary_datas = do_training_log(\
                            sess, train_summary_datas, now_epoch, netG_train_avg, data_run_time, sess_run_time)
                        stamp_index_TRAIN += 1
                        netG_train_avg = [0] * len(netG_train_summary_names)
                        netG_train_c_count = 0

                    if iter == LOG_STAMP_TEST[stamp_index_TEST]:
                        best_value_history, train_summary_datas = do_testing(now_epoch, data_loader, best_value_history, indices_1, indices_2, list_train_loss, \
                            summary_writer, train_summary_datas, True)
                        stamp_index_TEST += 1
                    total_iter = total_iter + 1
                if epoch >= FLAGS['netG_base_learning_decay_epoch']:
                    netG_base_lr = netG_base_lr - (FLAGS['netG_base_learning_rate'] / FLAGS['netG_base_learning_decay'])
                    print(current_time() + ', epoch = ' + now_epoch + ', new netG_base_lr = %g' % netG_base_lr)
                indices_1, indices_2 = data_loader.enqueue_train_indices_by_shuffle_method(indices_1, indices_2, 0)
        coord.request_stop()
        coord.join(threads)
        sess.close()
