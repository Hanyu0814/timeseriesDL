import numpy as np
import pickle
import json


class file_loader:
    def __init__(self, config_path = "data.json"):
        self.config = json.load(open(config_path, "r"))
        # how many timeslots per day (48 here)
        self.timeslot_daynum = int(86400 / self.config["timeslot_sec"])
        self.threshold = int(self.config["threshold"])
        self.isVolumeLoaded = False

    def load_volume(self):
        # shape (timeslot_num, x_num, y_num, type=2)
        self.volume = np.load(self.config["volume"]) / self.config["volume_max"]
        num_train = int(np.round(self.volume.shape[0] * 0.8))
        self.volume_train = self.volume[:num_train, :, :, :]
        self.volume_test = self.volume[num_train:, :, :, :]
        np.save('vol_test.npy',self.volume_test)
        self.isVolumeLoaded = True
    
    
    #this function nbhd for cnn, and features for lstm, based on attention model
    def sample_stdn(self, datatype, att_lstm_num = 3, long_term_lstm_seq_len = 3, short_term_lstm_seq_len = 7,\
    hist_feature_daynum = 7, last_feature_num = 48, nbhd_size = 1, cnn_nbhd_size = 3):
        if self.isVolumeLoaded is False:
            self.load_volume()

        if long_term_lstm_seq_len % 2 != 1:
            print("Att-lstm seq_len must be odd!")
            raise Exception

        if datatype == "train":
            data = self.volume_train
        elif datatype == "test":
            data = self.volume_test
        else:
            print("Please select **train** or **test**")
            raise Exception


        cnn_att_features = []
        lstm_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features.append([])
            cnn_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])

        cnn_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
        short_term_lstm_features = []
        labels = []

        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = data.shape[0]
        volume_type = data.shape[-1]

        for t in range(time_start, time_end):
            if t%100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    
                    #sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        #cnn features, zero_padding
                        cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

                        #lstm features
                        #nbhd (neighborhood) feature, zero_padding
                        nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                        #actual idx in data
                        for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                            for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                #boundary check
                                if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()

                        #last feature
                        last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                        #hist feature
                        hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    #sample att-lstms
                    for att_lstm_cnt in range(att_lstm_num):
                        
                        #sample lstm at att loc att_lstm_cnt
                        long_term_lstm_samples = []
                        # get time att_t, move forward for (att_lstm_num - att_lstm_cnt) day, then move back for ([long_term_lstm_seq_len / 2] + 1)
                        # notice that att_t-th timeslot will not be sampled in lstm
                        # e.g., **** (att_t - 3) **** (att_t - 2) (yesterday's t) **** (att_t - 1) **** (att_t) (this one will not be sampled)
                        # sample att-lstm with seq_len = 3
                        att_t = t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        #att-lstm seq len
                        for seqn in range(long_term_lstm_seq_len):
                            # real_t from (att_t - long_term_lstm_seq_len) to (att_t - 1)
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            #cnn features, zero_padding
                            cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                            #actual idx in data
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    # import ipdb; ipdb.set_trace()
                                    cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            #att-lstm features
                            # nbhd feature, zero_padding
                            nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                            #actual idx in data
                            for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                                for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                    #boundary check
                                    if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                        continue
                                    #get features
                                    nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()

                            #last feature
                            last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                            #hist feature
                            hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    #label
                    labels.append(data[t, x, y, :].flatten())


        output_cnn_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                output_cnn_att_features.append(cnn_att_features[i][j])
        
        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        labels = np.array(labels)

        return output_cnn_att_features, lstm_att_features, cnn_features, short_term_lstm_features, labels