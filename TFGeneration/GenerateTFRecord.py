import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import pickle
from tqdm import tqdm

class GenerateTFRecord:
    def __init__(self, inpath, outpath,filesize):
        self.outtfpath = outpath
        self.inpath=inpath
        self.filesize=filesize
        self.writer=None
        if(not os.path.exists(self.outtfpath)):
            os.mkdir(self.outtfpath)

        if (not os.path.exists(self.inpath)):
            print('\nInput directory does not exist')

        self.inpicklepath = inpath
        self.num_of_max_vertices=900
        self.max_length_of_word=30
        #self.str_to_chars=lambda str:np.chararray(list(str))

    def str_to_chars(self,str):
        charr=np.chararray(shape=(1,self.max_length_of_word))
        charr[:]=''
        charr[0,:len(str)]=list(str)
        return charr[0]

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tf_record(self, img_path, cellmatrix, rowmatrix, colmatrix, arr):

        # cellmatrix = cellmatrix.tostring()
        # colmatrix = colmatrix.tostring()
        # rowmatrix = rowmatrix.tostring()
        cellmatrix=self.pad_with_zeros(cellmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
        colmatrix = self.pad_with_zeros(colmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        rowmatrix = self.pad_with_zeros(rowmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))

        im = np.array(cv2.imread(img_path, 0),dtype=np.int64)
        img_height, img_width=im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)

        #words_arr = [val.encode('utf-8') for val in words_arr]
        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features=np.zeros(shape=(self.num_of_max_vertices,4),dtype=np.int64)
        vertex_features[:no_of_words,:]=arr[:,2:]

        #vertex_text=np.chararray(shape=(self.num_of_max_vertices,self.max_length_of_word))
        #vertex_text[:no_of_words,:]=list(map(self.str_to_chars, words_arr))
        vertex_text=words_arr+[""]*(self.num_of_max_vertices-len(words_arr))
        vertex_text = np.zeros(self.max_length_of_word, dtype=np.int64)

        # print("Max height", img_height)
        # print("Max width", img_width)
        # print("Len global features", 3)
        # print("Max vertices", self.num_of_max_vertices)
        # print("Data dims", 4)
        # print("Max words length", self.max_length_of_word)

        feature = dict()
        feature['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=im.astype(np.float32).flatten()))
        feature['global_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([img_height, img_width,no_of_words]).astype(np.float32).flatten()))
        feature['vertex_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=vertex_features.astype(np.float32).flatten()))

        feature['adjacency_matrix_cells'] = tf.train.Feature(int64_list=tf.train.Int64List(value=cellmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_cols'] = tf.train.Feature(int64_list=tf.train.Int64List(value=colmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_rows'] = tf.train.Feature(int64_list=tf.train.Int64List(value=rowmatrix.astype(np.int64).flatten()))
        feature['vertex_text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=vertex_text.astype(np.int64).flatten()))

        all_features = tf.train.Features(feature=feature)

        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def write_tf(self,input_files_paths,output_file_name):
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        with tf.python_io.TFRecordWriter(os.path.join(self.outtfpath,output_file_name),options=options) as writer:
            for img_path in input_files_paths:
                pickle_file = open(img_path.replace('.png', ''), 'rb')
                arr = pickle.load(pickle_file)

                colmatrix = np.array(arr[1],dtype=np.int64)
                cellmatrix = np.array(arr[2],dtype=np.int64)
                rowmatrix = np.array(arr[0],dtype=np.int64)
                bboxes = np.array(arr[3])
                seq_ex = self.generate_tf_record(img_path, cellmatrix, rowmatrix, colmatrix, bboxes)
                writer.write(seq_ex.SerializeToString())


    def write_to_tf(self):

        files_list=[]
        for file in os.listdir(self.inpicklepath)[:100]:
            if(file.endswith('.png')):
                files_list.append(os.path.join(self.inpicklepath,file))
        counter=1
        while(len(files_list)>self.filesize):
            self.write_tf(files_list[:self.filesize],str(counter)+'.tfrecord')
            files_list=files_list[self.filesize:]
            print('\nCompleted :', str(counter) + '.tfrecord')
            counter+=1

        if(len(files_list)>0):
            self.write_tf(files_list,str(counter)+'.tfrecord')
            print('\nCompleted :', str(counter) + '.tfrecord')
