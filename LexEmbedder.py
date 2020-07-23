#!/usr/bin/python3
# -*- coding: utf-8 -*-

## Author: Michi Amsler
## Date: 2018

import os

from gensim.models import Word2Vec, KeyedVectors

#### some utils directly in here:


import numpy as np
def get_most_similar_to_centroids(centroidfile=None, lexfile=None, embedding_model=None, topn=30, prettyprint = False):
    
    #load the embedding model
    #t0 = time()
    #embedding_model = load_embedding_model(embedding_model)
    #print("... loaded embeddings in %0.3fs." % (time() - t0))
    
    #read in the centroids from the file
    centroid_vectors = [line.strip().split("\t")[1:] for line in open(centroidfile)]
    
    #the labels from the file
    centroid_labels = [line.strip().split("\t")[0] for line in open(centroidfile)]
    
    #make an np array    
    centroids_array = np.array(centroid_vectors, dtype=np.float32)
    
    ms_words = []
    lex_words = read_in_lexicon(lexfile)
    
    t0 = time()
    #unbound here  ...
    for centroid in centroids_array[0:]:
        print("most similar to {}:".format(centroid_labels[np.nonzero(centroids_array==centroid)[0][0]]))    
        #give 20 most similar
        most_similars = embedding_model.similar_by_vector(centroid, topn=topn)
        
        if prettyprint:
            print("\n".join(["\t".join((str(el) for el in tup)) for tup in most_similars]))        
        else:
            print(most_similars)
        print("\n")
        
        #print(embedding_model.similar_by_vector(centroid, topn=topn)) 
        
        for word, dist in most_similars:
            ms_words.append(word)
    
    
    not_in_lex = [w for w in ms_words if w not in lex_words]
    
    print("words in ms_words: {}".format(len(ms_words)))
    print("words in set(lex_words): {}".format(len(set(lex_words))))
    
    print("not in lexicon:")
    print(not_in_lex)
    
    print("... computed most similar to centroids in %0.3fs." % (time() - t0))
    

def lexicons_to_cluster(list_of_lexicon_files=None, 
                        list_of_prefixes=None, 
                        embedding_model=None,
                        outfilename = "CLUSTERED_lex.TXT",
                        target_folder = ".",
                        write_out = True,
                        number_of_clusters=10, 
                        number_of_n_init=100, 
                        number_of_max_iter=1000):
    '''
    takes 
        - a list of lexicons (with paths)
        - OPTIONAL a list of prefix_names  (n prefixes for n_lexicons)
        - an embedding model
        - an outfilename (do provide!)
        - OPTIONAL number of clusters (default 10)
        - OPTIONAL number of inits (default 100)
        - OPTIONAL number of max iterations (default 1000)
    '''
    
    do_clustering(      list_of_lexicon_files=list_of_lexicon_files, 
                        list_of_prefixes=list_of_prefixes, 
                        embedding_model=embedding_model,
                        outfilename = outfilename,
                        target_folder = target_folder,
                        write_out = write_out, 
                        number_of_clusters=number_of_clusters, 
                        number_of_n_init=number_of_n_init, 
                        number_of_max_iter=number_of_max_iter)
    
    
    return

    
def read_in_lexicon(lexicon_filename):
    '''
    load 1 lexicon.
    read in; filter; put into list; return list of lexicon entries
    '''
    
    with open(lexicon_filename,"r") as lex_file:
        #starting ...
        print("loading lexicon {}".format(lexicon_filename))
        #where we put them
        lex_words = []
        #iteration
        for line in lex_file:
            #commments and empty ones: skip aka. continue
            if line.startswith("#") or line.strip() == "":
                continue
            else:
                #get the word
                lex_words.append(line.strip())
        #done; count; report
        print("got {} entries".format(len(lex_words)))    
        print("done")
        
    # a list of lexicon word
    return lex_words


def get_embeddings_for_lex_entries(lexlist, emb_model, lowercase_embedding = False):
    '''
    embeds the lexicon list
    i.e. we get the produce a dictionary with the words as keys
    and the as values we get the x-dim_list
    '''
    
    corr_counter = 0 
    counter = 0 
    res_dict = {}
    
    if lowercase_embedding:
        lexlist = [entry.lower() for entry in lexlist]
    
    for word in lexlist:
            #print("getting embeddings for {}".format(word))
            try:
                feature_400dim_list = list(emb_model[word])
                res_dict[word] = feature_400dim_list
                corr_counter+=1
            except KeyError :
                #print("got no representation for {}".format(word))
                counter+=1
                continue 

    print("got representations for {} words".format(corr_counter))
    print("got no representation for {} words".format(counter))
    
    return res_dict



import os.path 
from time import time
from sklearn.cluster import KMeans
def do_clustering(  list_of_lexicon_files=None, 
                    list_of_prefixes=None, 
                    embedding_model=None,
                    outfilename = "CLUSTERED_lex.TXT",
                    target_folder = ".",
                    write_out = True,
                    number_of_clusters=10, 
                    number_of_n_init=100, 
                    number_of_max_iter=1000):
    
    #########################################################################

    
    #set up clustering:
    k_means = KMeans(   init='k-means++', 
                    n_clusters=number_of_clusters, 
                    n_init=number_of_n_init,
                    max_iter=number_of_max_iter,
                    #uses 3 cores
                    verbose=1, 
                    n_jobs=-2)
    
    
    
    #########################################################################
    
    #load embeddings:
    
    #load the model
    #t0 = time()
    #embedding_model = load_embedding_model(embedding_model)
    #print("... loaded embeddings in %0.3fs." % (time() - t0))
    
    
    
    
    
    #open output file so that we can write per lex_file:
    with open(os.path.join(target_folder, outfilename), "w", encoding="utf-8") as centroid_file:
        print("writing to {}".format(outfilename))
        #read in and put into store
        for index_names, lex_file in enumerate(list_of_lexicon_files):
            #
            print("reading in {}".format(lex_file))
            filename, extension = os.path.splitext(os.path.split(lex_file)[1])
            
            #print(filename, extension)
            
            
            #store
            words = []
            datapoints_words = []
            # open file; read in words and ve
            lexicon_word_list = read_in_lexicon(lex_file)
            # the get embeddings for this list:
            word_emb_dict =  get_embeddings_for_lex_entries(lexicon_word_list, embedding_model)
            
            for (w, emblist) in sorted(word_emb_dict.items()):
                words.append(w)
                datapoints_words.append([float(el) for el in emblist])
            
            #do it:
            print("we have {} words(entries) embedded".format(len(datapoints_words)))
            print("we have {} dimensions".format(len(datapoints_words[0])))
           
            X = datapoints_words
            print("starting clustering with:\n{} clusters\t{} initialisations\t{} max iterations".format(number_of_clusters, number_of_n_init, number_of_max_iter))
            print("start clustering ...")
            t0 = time()
            k_means.fit(X)
            print("... clustered in %0.3fs." % (time() - t0))

            #with codecs.open(outfilename, "a", "utf-8") as centroid_file:
            for index, cluster_centroid_vec in enumerate(k_means.cluster_centers_):
                #name = lexname_centroidX
                if list_of_prefixes is not None:
                    name_of_centroid = list_of_prefixes[index_names] + "_" + str(index)
                else:
                    name_of_centroid = filename + "_"+ str(index)
                
                if write_out:
                    centroid_file.write(name_of_centroid + "\t" + "\t".join((str(el) for el in cluster_centroid_vec))+"\n")
    

def ad_hoc_clustering( list_of_words_to_cluster = None,
                    embedding_model=None,
                    lowercase_embedding = False,
                    number_of_clusters=10, 
                    number_of_n_init=100, 
                    number_of_max_iter=1000,
                    topn = 20,
                    uniq=True,
                    prettyprint = False, 
                    verbosity=0):
    
    #########################################################################

    
    #set up clustering:
    k_means = KMeans(   init='k-means++', 
                    n_clusters=number_of_clusters, 
                    n_init=number_of_n_init,
                    max_iter=number_of_max_iter,
                    #uses 3 cores
                    verbose=verbosity, 
                    n_jobs=-2)

    #store
    words = []
    datapoints_words = []
    
    if lowercase_embedding:
        list_of_words_to_cluster = [w.lower() for w in list_of_words_to_cluster]
    
    
    # the get embeddings for this list:
    word_emb_dict =  get_embeddings_for_lex_entries(list_of_words_to_cluster, embedding_model)
    
    #fill the lists; same indices; sorted
    if uniq:
        for (w, emblist) in sorted(word_emb_dict.items()):
            words.append(w)
            datapoints_words.append([float(el) for el in emblist])
    #this version is for when we want to give in the weights of words via number of occurence in the list_of_words_to_cluster
    else:
        for word in sorted(list_of_words_to_cluster):
            words.append(word)
            datapoints_words.append([float(el) for el in word_emb_dict[word]])
                
    #do it:
    print("we have {} words(entries) embedded".format(len(datapoints_words)))
    print("we have {} dimensions".format(len(datapoints_words[0])))
           
    X = datapoints_words
    print("starting clustering with:\n{} clusters\t{} initialisations\t{} max iterations".format(number_of_clusters, number_of_n_init, number_of_max_iter))
    print("start clustering ...")
    t0 = time()
    k_means.fit(X)
    print("... clustered in %0.3fs." % (time() - t0))

    #make an np array with the centroids   
    centroids_array = np.array(k_means.cluster_centers_, dtype=np.float32)
    
    most_similar_words = []
    #lex_words = list_of_words_to_cluster

    t0 = time()
    #get most similars to centroids:
    for index, centroid in enumerate(centroids_array):
        print("most similar to CENTROID {}:".format(index))    
        #give 20 most similar
        most_similars = embedding_model.similar_by_vector(centroid, topn=topn)
        
        if prettyprint:
            print("\n".join(["\t".join((str(el) for el in tup)) for tup in most_similars]))        
        else:
            print(most_similars)
        print("\n")
        
        #print(embedding_model.similar_by_vector(centroid, topn=topn)) 
        
        for word, dist in most_similars:
            most_similar_words.append(word)
    
    
    not_in_lex = set([w for w in most_similar_words if w not in list_of_words_to_cluster])
    
    not_in_most_similars = set([w for w in list_of_words_to_cluster if w not in most_similar_words])
    
    print("words in most_similar_words: {}".format(len(most_similar_words)))
    print("words in set(list_of_words_to_cluster): {}".format(len(set(list_of_words_to_cluster))))
    
    print("not in lexicon:")
    print(not_in_lex)
    
    print("in lex but not in the vicinity of centroids (given topn):")
    print(not_in_most_similars)
    
    print("... computed most similar to centroids in %0.3fs." % (time() - t0))

    return centroids_array
    
def show_centroid_words(list_of_words_clustered = None,
                    embedding_model=None,
                    centroids_list = None,
                    topn = 20,
                    prettyprint = False,
                    threshold = None
                    ):
    
    most_similar_words = []
    #lex_words = list_of_words_to_cluster

    t0 = time()
    #get most similars to centroids:
    for index, centroid in enumerate(centroids_list):
        print("most similar to CENTROID {}:".format(index))    
        #give 20 most similar
        most_similars = embedding_model.similar_by_vector(centroid, topn=topn)
        
        if prettyprint:
            print("\n".join(["\t".join((str(el) for el in tup)) for tup in most_similars]))        
        else:
            print(most_similars)
        print("\n")
        
        #print(embedding_model.similar_by_vector(centroid, topn=topn)) 
        
        
        for word, sim in most_similars:
            if threshold is not None:
                #apply threshold
                if sim >= threshold:  
                    most_similar_words.append(word)
                else:
                    continue
            else:
                most_similar_words.append(word)
    
    
    not_in_lex = set([w for w in most_similar_words if w not in list_of_words_clustered])
    
    not_in_most_similars = set([w for w in list_of_words_clustered if w not in most_similar_words])
    
    print("words in most_similar_words: {}".format(len(most_similar_words)))
    print("words in set(list_of_words_clustered): {}".format(len(set(list_of_words_clustered))))
    
    print("not in lexicon:")
    print(not_in_lex)
    
    print("in lex but not in the vicinity of centroids (given topn):")
    print(not_in_most_similars)
    
    print("... computed most similar to centroids in %0.3fs." % (time() - t0))
    
    return most_similar_words
    
    
    #########################################################################
####  end utils ################################

class LexEmbedder(object):
    """takes lexicons and embeds them
    then clusters them and returns the centroids
    output those centroids in a file with "name"
    GIVEN_LEX_FILE_[INDEX_of_CENTROID]
    """

    def __init__(self, 
                    given_embedding_file = None,
                    embedding_model = None, # use already loaded ...
                    list_of_given_lexicon_files = None,
                    list_of_given_prefixes = None,
                    list_of_given_outnames = None,
                    given_target_folder = ".",
                    lex_folder = None,
                    ):
        
        self.given_embedding_file = given_embedding_file
        self.embedding_model = embedding_model

        self.target_folder = given_target_folder
        self.lex_folder = lex_folder

        self.list_of_lexicons  = list_of_given_lexicon_files
        self.list_of_prefixes  = list_of_given_prefixes
        self.list_of_outnames  = list_of_given_outnames

    def load_lexicons(self, given_lexicon_file_list = None):
        
        #this is the case for an overwrite or ex-post change
        if given_lexicon_file_list is not None:
            self.list_of_lexicons = given_lexicon_file_list


        use_generic_prefixes = False
        use_generic_outnames = False

        # we need to create these empy lists here
        if self.list_of_prefixes is None:
            self.list_of_prefixes = list()
            use_generic_prefixes = True
        if self.list_of_outnames is None:
            self.list_of_outnames = list()
            use_generic_outnames = True


        for lexfile in self.list_of_lexicons:
            
            lex_path, lex_filename = os.path.split(lexfile)
            name, extension = os.path.splitext(lex_filename)

            # a bit silly: actually not needed to re-insert that ...
            #curr_lexicon = os.path.join(lex_path, lex_filename), 
            
            curr_prefix = name 
            curr_outname = os.path.join(self.target_folder, name + ".centroids")

            #self.list_of_lexicons.append(curr_lexicon)
            
            if use_generic_prefixes:
                self.list_of_prefixes.append(curr_prefix)
            #else: use those lists; we don't do anything here then ...


            if use_generic_outnames:
                #print("append {} to outnames".format(curr_outname))
                self.list_of_outnames.append(curr_outname)    
            #else: use those lists; we don't do anything here then ...

        return
    
    def load_lexicons_from_folder(self, given_folder_with_lexicons = None):
        """here we get a folder with lexicons (basically files with .txt extension)
        we iterate over them and build a list of
        - lexfiles
        - prefixes (one per lexfile)
        - outnames (one per lexfile)

        those lists have a "connected" index to use the information easily
        however, we do not save it as a tuple since we may want to change some 
        elements during processing        

        Keyword Arguments:
            given_folder_with_lexicons {[type]} -- [description] (default: {None})
        """

        #this is the case for an overwrite or ex-post change
        if given_folder_with_lexicons is not None:
            self.lex_folder = given_folder_with_lexicons

        # creating the lists:
        self.list_of_lexicons = list()
        self.list_of_prefixes = list()
        self.list_of_outnames = list()


        with os.scandir(self.lex_folder) as it:
            # we use all .txt files in the given folder
            for entry in it:
                if entry.name.endswith('.txt') and entry.is_file():
                    name, extension = os.path.splitext(entry.name)
                    
                    # assembling the names, prefixes and outnames
                    curr_lexicon = os.path.join(self.lex_folder, entry.name)
                    curr_prefix = name
                    curr_outname = os.path.join(self.target_folder, name + ".centroids")

                    #appending this to the lists
                    self.list_of_lexicons.append(curr_lexicon)
                    self.list_of_prefixes.append(curr_prefix)
                    self.list_of_outnames.append(curr_outname)


        return

    def load_embeddings(self, given_model_name = None, mode = "w2v"):
        """wrapper for embedding loader
        """

        #check if we have an overwrite:
        if given_model_name is not None:
            model_file_to_read_from = given_model_name
        else:
            model_file_to_read_from = self.given_embedding_file


        if mode == "w2v":
            self.load_w2v_model(given_model_name=model_file_to_read_from )
        else:
            print("not yet implemented!")
        
        return

    def load_w2v_model(self, given_model_name = None):
        
        '''load models; simple wrapper'''

        

        t0 = time()

        print("loading model {} ...".format(given_model_name))
        try:
            self.embedding_model = Word2Vec.load(given_model_name, mmap="r")
        except:
            print("trying loading with keyedvectors method")
            self.embedding_model = KeyedVectors.load(given_model_name, mmap="r")

        print("... done in %0.3fs." % (time() - t0))

        return

    def prepare(self, verbose = False):
        
        print("checking targetfolder")
        #check if targetfolder exists; if not create:
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
            print("{} created".format(self.target_folder))

        #check if we have a given embedding model: then use it
        # if not: check if we were provided with a path to a model
        # if so: try to load this one:
        #load if nothing is provided
        print("setting up embeddings ...")
        if self.embedding_model is None:
            try:
                print("loading embeddings")
                self.load_embeddings()
            except:
                if self.given_embedding_file is None:
                    print("please add an embedding model!")
        else:
            print("using passed model {}".format(self.embedding_model))


        # load lexfile(s)
        #first: were we given a lex_source_folder?
        if self.lex_folder is not None:
            self.load_lexicons_from_folder()
        else:
            #we suspect that we were then given a list of lexfiles:
            #try:
            self.load_lexicons()
            #except:
            #    print("couldn't load any lexicons!")
       
        if verbose:
            self.show_config()


        return

    def show_config(self):
        """show brief summary of config of the expander:
        """
        print (""" 
        list_of_lexicons: {}
        list_of_prefixes: {}
        list_of_outnames: {}
        target_folder: {}
        lex_folder: {}
        embedding_model: {}
        given_embedding_file: {}
        """.format(self.list_of_lexicons,
        self.list_of_prefixes, 
        self.list_of_outnames, 
        self.target_folder, 
        self.lex_folder , 
        self.embedding_model, 
        self.given_embedding_file, 
        ))
        return


    def run(self):
        
        self.embed_lexicons()

        return



#####

    def embed_lexicons(self, number_of_clusters_given = 10 ,write_out = True):
        
        print("we embed the following lexicons: {}".format(self.list_of_lexicons))

        for index, lexfile in enumerate(self.list_of_lexicons):
    
            print("clustering lexiconfile {} to centroidfile {}, using the prefix {} for the clustercentroid names".format(lexfile, self.list_of_outnames[index], self.list_of_prefixes[index]))
    
            lexicons_to_cluster(list_of_lexicon_files=[lexfile],
                        list_of_prefixes=[self.list_of_prefixes[index]], 
                        embedding_model=self.embedding_model,
                        outfilename = self.list_of_outnames[index],
                        target_folder = self.target_folder, 
                        write_out = write_out,
                        number_of_clusters=number_of_clusters_given, 
                        number_of_n_init=1000, 
                        number_of_max_iter=10000)

            #this gives some output showing where the centroids have been placed (by showing most_similars)
            get_most_similar_to_centroids(  centroidfile = os.path.join(self.target_folder, self.list_of_outnames[index]), 
                                            lexfile = lexfile, 
                                            embedding_model = self.embedding_model, 
                                            topn=10)

