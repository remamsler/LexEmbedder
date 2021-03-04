#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import:
import LexEmbedder


# In[2]:


#create LexEmbedder instance
# we also need an embedding 
# and we give it a list of lexicons for which we want centroids:

EMBEDDINGFILE = "smd_50k_kv"

my_embedder = LexEmbedder.LexEmbedder(
                                    given_embedding_file = "smd_50k_kv", # re-use; no loading required 
                                    list_of_given_lexicon_files = ["tiere_not_so_minimal.txt"] ,
                                    list_of_given_prefixes = ["animal"],
                                    list_of_given_outnames = ["animal.centroids"],
                                    )


# In[3]:


#prepare
my_embedder.prepare(verbose=True)


# In[4]:


#shoot of
my_embedder.run()


# In[ ]:


#reload and control

