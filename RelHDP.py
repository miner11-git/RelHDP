#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy
from scipy.special import gammaln
import logging
import gensim
from itertools import combinations
import sys
from scipy import special

import math
numpy.random.seed(10)
from collections import Counter

import pandas as pd
import numpy as np
import xlrd
import collections
import random




class DefaultDict(dict):
    def __init__(self, v):
        self.v = v
        dict.__init__(self)

    def __getitem__(self, k): 
        return dict.__getitem__(self, k) if k in self else self.v

    def update(self, d):  
        dict.update(self, d)
        return self


class HDPLDA:
    def __init__(self, alpha, beta, gamma, docs, V):
        self.alpha = alpha  
        self.beta = beta 
        self.gamma = gamma 
        self.V = V
        self.M = len(docs)

        
        self.using_t = [[0] for j in range(self.M)]

       
        self.using_k = [0]

        self.x_ji = docs  
        self.k_jt = [numpy.zeros(1, dtype=int) for j in range(self.M)]  
        self.n_jt = [numpy.zeros(1, dtype=int) for j in range(self.M)]  
        self.n_jtv = [[None] for j in range(self.M)]
        self.w_jt = [[[]] for j in range(self.M)]  

        self.m = 0  
        self.m_k = numpy.ones(1, dtype=int)  
        self.n_k = numpy.array([self.beta * self.V])  
        self.n_kv = [DefaultDict(0)]  

        self.t_ji = [numpy.zeros(len(x_i) - 4, dtype=int) - 1 for x_i in docs]

    def inference(self):       
        for j, x_i in enumerate(self.x_ji):
            x_i = x_i[4:]  
            for i in range(len(x_i)):
                self.sampling_t(j, i)
        for j in range(self.M):
            for t in self.using_t[j]:
                if t != 0: self.sampling_k(j, t)

    def worddist(self):
        
        return [DefaultDict(self.beta / self.n_k[k]).update(
            (v, n_kv / self.n_k[k]) for v, n_kv in self.n_kv[k].items())
            for k in self.using_k if k != 0]

    def docdist(self):
        
        am_k = numpy.array(self.m_k, dtype=float)
        am_k[0] = self.gamma  
        am_k *= self.alpha / am_k[self.using_k].sum()  

        theta = []
        for j, n_jt in enumerate(self.n_jt):
            p_jk = am_k.copy()
            for t in self.using_t[j]:
                if t == 0: continue
                k = self.k_jt[j][t]
                p_jk[k] += n_jt[t]  
            p_jk = p_jk[self.using_k]
            theta.append(p_jk / p_jk.sum())

        return numpy.array(theta)

    def perplexity(self):
        
        phi = [DefaultDict(1.0 / self.V)] + self.worddist()
        theta = self.docdist()
        log_likelihood = 0
        N = 0
        for x_ji, p_jk in zip(self.x_ji, theta):
            x_ji = x_ji[4:]  
            for v in x_ji:
                word_prob = sum(p * p_kv[v] for p, p_kv in zip(p_jk, phi))
                log_likelihood -= numpy.log(word_prob)
            N += len(x_ji)
        return numpy.exp(log_likelihood / N)

    def complexity(self):  
        C = len(self.using_k) - 1  
        for k_jt in self.k_jt:
            C += len(set(k_jt))  
        return C

    def dump(self, disp_x=False):
        if disp_x: print("x_ji:", self.x_ji)
        print("using_t:", self.using_t)
        print("t_ji:", self.t_ji)
        print("using_k:", self.using_k)
        print("k_jt:", self.k_jt)
        print("----")
        print("n_jt:", self.n_jt)
        print("n_jtv:", self.n_jtv)
        print("w_jt:", self.w_jt)  
        print("n_k:", self.n_k)
        print("n_kv:", self.n_kv)
        print("m:", self.m)
        print("m_k:", self.m_k)
        print(" ")

    def sampling_t(self, j, i): 
        self.leave_from_table(j, i)

        v = self.x_ji[j][4:][i]  
        f_k = self.calc_f_k(v)  
        assert f_k[0] == 0  

        
        p_t = self.calc_table_posterior(j, f_k)
        if len(p_t) > 1 and p_t[1] < 0: self.dump()
        t_new = self.using_t[j][numpy.random.multinomial(1, p_t).argmax()]  
        if t_new == 0: 
            p_k = self.calc_dish_posterior_w(j, f_k)
            k_new = self.using_k[numpy.random.multinomial(1, p_k).argmax()]  
            if k_new == 0:  
                k_new = self.add_new_dish()
            t_new = self.add_new_table(j, k_new)

       
        self.seat_at_table(j, i, t_new)

    def leave_from_table(self, j, i):
        t = self.t_ji[j][i]  
        if t > 0:
            k = self.k_jt[j][t]  
            assert k > 0

            
            v = self.x_ji[j][4:][i]  
            self.n_kv[k][v] -= 1
            self.n_k[k] -= 1  
            self.n_jt[j][t] -= 1  
            self.n_jtv[j][t][v] -= 1
            self.w_jt[j][t].remove(v)  

            if self.n_jt[j][t] == 0:  
                self.remove_table(j, t)

    def remove_table(self, j, t):
       
        k = self.k_jt[j][t]  
        self.using_t[j].remove(t)  
        self.m_k[k] -= 1  
        self.m -= 1  
        assert self.m_k[k] >= 0
        if self.m_k[k] == 0:  
            
            self.using_k.remove(k)

    def calc_f_k(self, v):  
        
        return [n_kv[v] for n_kv in self.n_kv] / self.n_k  

    def calc_table_posterior(self, j, f_k):
        using_t = self.using_t[j]
        p_t = self.n_jt[j][using_t] * f_k[self.k_jt[j][using_t]]  
        p_x_ji = numpy.inner(self.m_k,
                             f_k) + self.gamma / self.V  
        p_t[0] = p_x_ji * self.alpha / (self.gamma + self.m)  
        return p_t / p_t.sum()

    def seat_at_table(self, j, i, t_new):
        assert t_new in self.using_t[j]
        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += 1

        k_new = self.k_jt[j][t_new]
        self.n_k[k_new] += 1

        v = self.x_ji[j][4:][i]  
        self.n_kv[k_new][v] += 1
        self.n_jtv[j][t_new][v] += 1
        self.w_jt[j][t_new].append(v)  

    
    def add_new_table(self, j, k_new):
        assert k_new in self.using_k
        for t_new, t in enumerate(self.using_t[j]):
            if t_new != t: break
        else:  
            t_new = len(self.using_t[j])
            self.n_jt[j].resize(t_new + 1)
            self.k_jt[j].resize(t_new + 1)
            self.n_jtv[j].append(None)
            self.w_jt[j].append([])  

        self.using_t[j].insert(t_new, t_new)
        self.n_jt[j][t_new] = 0 
        self.n_jtv[j][t_new] = DefaultDict(0)
        self.w_jt[j][t_new] = []  

        self.k_jt[j][t_new] = k_new
        self.m_k[k_new] += 1
        self.m += 1

        return t_new

    def calc_dish_posterior_w(self, j, f_k):  
       
        p_k = (self.m_k * f_k)[self.using_k]  
        p_k[0] = self.gamma / self.V  
        return p_k / p_k.sum()

    def sampling_k(self, j, t):  
        
        self.leave_from_dish(j, t)

        
        p_k = self.calc_dish_posterior_t(j, t)
        k_new = self.using_k[numpy.random.multinomial(1, p_k).argmax()]
        if k_new == 0:  
            k_new = self.add_new_dish()

        self.seat_at_dish(j, t, k_new)

    def leave_from_dish(self, j, t):
        
        k = self.k_jt[j][t]
        assert k > 0
        assert self.m_k[k] > 0
        self.m_k[k] -= 1
        self.m -= 1
        if self.m_k[k] == 0:  
            self.using_k.remove(k)
            self.k_jt[j][t] = 0

    def calc_dish_posterior_t(self, j, t):  
        k_old = self.k_jt[j][t]  
        
        Vbeta = self.V * self.beta
        n_k = self.n_k.copy()
        n_jt = self.n_jt[j][t]
        n_k[k_old] -= n_jt  
        n_k = n_k[self.using_k]
        log_p_k = numpy.log(self.m_k[self.using_k]) + gammaln(n_k) - gammaln(n_k + n_jt)  
        log_p_k_new = numpy.log(self.gamma) + gammaln(Vbeta) - gammaln(Vbeta + n_jt)
        

        gammaln_beta = gammaln(self.beta)
        for w, n_jtw in self.n_jtv[j][t].items():
            assert n_jtw >= 0
            if n_jtw == 0: continue
            n_kw = numpy.array([n.get(w, self.beta) for n in self.n_kv])  
            n_kw[k_old] -= n_jtw  
            n_kw = n_kw[self.using_k]
            n_kw[0] = 1  
            if numpy.any(n_kw <= 0): print(n_kw)  
            log_p_k += gammaln(n_kw + n_jtw) - gammaln(n_kw)
            
            log_p_k_new += gammaln(self.beta + n_jtw) - gammaln_beta
            
        log_p_k[0] = log_p_k_new
        
        p_k = numpy.exp(log_p_k - log_p_k.max())
        return p_k / p_k.sum()

    def seat_at_dish(self, j, t, k_new):
        self.m += 1
        self.m_k[k_new] += 1

        k_old = self.k_jt[j][t]  
        if k_new != k_old:
            self.k_jt[j][t] = k_new

            n_jt = self.n_jt[j][t]
            if k_old != 0: self.n_k[k_old] -= n_jt
            self.n_k[k_new] += n_jt
            for v, n in self.n_jtv[j][t].items():
                if k_old != 0: self.n_kv[k_old][v] -= n
                self.n_kv[k_new][v] += n

    def add_new_dish(self):
        
        for k_new, k in enumerate(self.using_k):
            if k_new != k: break
        else: 
            k_new = len(self.using_k)
            if k_new >= len(self.n_kv):
                self.n_k = numpy.resize(self.n_k, k_new + 1)
                self.m_k = numpy.resize(self.m_k, k_new + 1)
                self.n_kv.append(None)
            assert k_new == self.using_k[-1] + 1
            assert k_new < len(self.n_kv)

        self.using_k.insert(k_new, k_new)
        self.n_k[k_new] = self.beta * self.V
        self.m_k[k_new] = 0
        self.n_kv[k_new] = DefaultDict(self.beta)
        return k_new


def hdplda_learning(hdplda, iteration):
    stopIterationThreshold = 10.0
    pre_perp = float(sys.maxsize)
    for i in range(iteration):
        hdplda.inference()
        perp = hdplda.perplexity()      
        
        if pre_perp:
            if abs(pre_perp - perp) <= stopIterationThreshold:  
                return hdplda
            else:
                pre_perp = perp
    return hdplda

def get_word_for_jt(hdplda):
    topic_num=max(hdplda.using_k)
    wn_topic=[[[] for k in range(hdplda.M)] for j in range(topic_num)]
    for j in range(hdplda.M):
        for t in range(len(hdplda.k_jt[j])):
            if hdplda.k_jt[j][t]==0:
                t=t+1
            else:
                t_num=hdplda.k_jt[j][t]
                for w in hdplda.w_jt[j][t]:
                    wn_topic[t_num-1][j].append(w)
    
    w_topic=[[] for k in range(topic_num)]
    for i in range(len(w_topic)):
        w_topic[i]=list(filter(None,wn_topic[i]))

    w_topic1=list(filter(None,w_topic))

    return w_topic1


def hHDP(hdplda,level,num,len_id):
    output_summary(hdplda)
    word_topic=get_word_for_jt(hdplda)
    topic_n= len(word_topic)
    
    hHDP_tree = [[] for t in range(level)]
    node_num = [[] for t in range(level+1)]
    node_num[0].append(1)
    print("***LEVEL 0***\n")

    t_w=[]
    for i in range(len(word_topic)):
        t_w.append(sum(word_topic[i],[]))

    freq_word_for_k=[]
    for j in range(len(t_w)):
        tmp=[n[0] for n in Counter(t_w[j]).most_common(100)]
        freq_word_for_k.append(tmp)
    node_topic = sum(freq_word_for_k, [])
    
    hHDP_tree[0].append(node_topic)
    print_t = [i[0] for i in Counter(node_topic).most_common(num)]
    print('NODE 1:', print_t)
    tmp_tree = word_topic[0:]
    node_num[1].append(len(tmp_tree))
    alpha = 1.5
    beta = 0.01  
    gamma = 0.5  

    
    def expand_hHDP_tree_by_HDP(tmp_tree, hHDP_tree, node_num, i, it,len_id):

        if i==1:
            tmp_tree_1=[]
            j = 0
            class_id = 0
            while j < it:
                if len(tmp_tree) == 0:
                    break
                docs = []
                for word_list in tmp_tree[0]:
                    print('word_list=', word_list)
                 
                    temp=random.randint(1,30)
                    class_id_str=str(temp)
                    prefix=['0','0',class_id_str,'0']
                    doc=prefix+word_list
                    docs.append([v for v in doc])
                hdplda1=HDPLDA(alpha,beta,gamma,docs,len_id)
                hdplda1=hdplda_learning(hdplda1,100)
                output_summary(hdplda1)

                wn_topic1 = get_word_for_jt(hdplda1)
                t_w = []  
                for k in range(len(wn_topic1)):
                    t_w.append(sum(wn_topic1[k], []))  

                freq_word_for_k = []  
                for m in range(len(t_w)):
                    tmp = [n[0] for n in Counter(t_w[m]).most_common(100)]
                    freq_word_for_k.append(tmp)
                node_topic1 = sum(freq_word_for_k, [])
                
                hHDP_tree[i].append(node_topic1)
                tmp_tree.remove(tmp_tree[0])
                tmp_tree_1.append(tmp_tree[0])
                print_t = [i[0] for i in Counter(node_topic1).most_common(num)]
                
                print('NODE', j + 1, ":", print_t)
                logger.info('NODE %d : %s' % ( j + 1, str(print_t)))
                if wn_topic1[0:] != []: tmp_tree.extend(wn_topic1[0:])
                
                node_num[i + 1].append(len(wn_topic1[0:]))

                
                j += 1
            tmp_tree=tmp_tree_1
        else:
            j = 0
            class_id = 0  
            while j < it:  
                if len(tmp_tree) == 0:
                    break
                docs = []
                for word_list in tmp_tree[0]:
                    
                    print('word_list=', word_list)
                    
                    temp = random.randint(1, 30)  
                    class_id_str = str(temp)
                    prefix = ['0', '0', class_id_str, '0']
                    doc = prefix + word_list
                    docs.append([v for v in doc])
                hdplda1 = HDPLDA(alpha, beta, gamma, docs, len_id)
                hdplda1 = hdplda_learning(hdplda1, 100)
                output_summary(hdplda1)
                phi = hdplda1.worddist()
                for k, phi_k in enumerate(phi):
                    print_t = []
                    for w in sorted(phi_k, key=lambda w: -phi_k[w])[:1000]:
                        
                        print_t.append(w)
                    j =j+1
                    print('NODE', j, ":", print_t)
                    logger.info('NODE %d : %s' % (j + 1, str(print_t)))
                    hHDP_tree[i].append(print_t)
                tmp_tree.remove(tmp_tree[0])



    for i in range(1, level):
        print(' ')
        print("***LEVEL %d***" % i)
        
        it = sum(node_num[i])       
        expand_hHDP_tree_by_HDP(tmp_tree, hHDP_tree, node_num, i, it,len_id)#0106内外都用PSHDP取生成

    print("node_num ：",node_num[:level])
    return hHDP_tree, node_num[:level]


def cosine_similarity(vector1, vector2):
  dot_product = 0.0
  normA = 0.0
  normB = 0.0
  for a, b in zip(vector1, vector2):
    dot_product += a * b
    normA += a ** 2
    normB += b ** 2
  if normA == 0.0 or normB == 0.0:
    return 0
  else:
    return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)

def output_summary(hdplda, minWordCountInTopic4KL=10):
    K = len(hdplda.using_k) - 1  
    logger.info('\n-- dynamically select %d topics.' % K)
    kmap = dict((k, i - 1) for i, k in enumerate(hdplda.using_k))
    dishcount = numpy.zeros(K, dtype=int)
    wordcount = [DefaultDict(0) for k in range(K)]
    for j, x_ji in enumerate(hdplda.x_ji):
        x_ji = x_ji[4:]  
        for v, t in zip(x_ji, hdplda.t_ji[j]):
            k = kmap[hdplda.k_jt[j][t]]
            dishcount[k] += 1
            wordcount[k][v] += 1

    
    phi = hdplda.worddist()
    print('phi=',phi)
    for k, phi_k in enumerate(phi):
        for w in sorted(phi_k, key=lambda w: -phi_k[w])[:1000]:  
            logger.info("%s: %f (%d)" % (w, phi_k[w], wordcount[k].get(w, 0)))

    if K <= 1:
        return
    keyWord4topic = []
    for k in range(K):
        keyWord4topic.append(
            sorted(phi[k], key=lambda w: -phi[k][w])[:100])  
    KL_topic = []
    KL_index = []
    COS=[]
    for z1, z2 in combinations(range(K), 2):
        KL = 0
        if (dishcount[z1] < minWordCountInTopic4KL) or (dishcount[z2] < minWordCountInTopic4KL):
            continue
        upper=[]
        lower=[]
        for v in set(keyWord4topic[z1] + keyWord4topic[z2]):  
            KL += phi[z1][v] * numpy.log2(phi[z1][v] / phi[z2][v])
            upper.append(phi[z1][v])
            lower.append(phi[z2][v])
        KL_topic.append(KL)
        KL_index.append((z1, z2))
        cos_sim=cosine_similarity(upper,lower)
        COS.append(cos_sim)
    if len(KL_topic) == 0:
        return
    
def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha",
                      default=1.5)  
    parser.add_option("--gamma", dest="gamma", type="float", help="parameter gamma",
                      default=0.01)  
    parser.add_option("--beta", dest="beta", type="float", help="parameter of beta measure H", default=0.5)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="stopwords", type="int", help="0=exclude stop words, 1=include stop words", default=1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    
    if options.seed != None:
        numpy.random.seed(options.seed)

    
    train = []
    text = []
    fp = open(filename)  
    for line in fp:
        line = line.split()
        text.append([w for w in line])
    for n in range(0, len(text)):
        train.append((text[n][4:]))

    id2wordDict = gensim.corpora.Dictionary(train)
   
    docs = []
    timeDict = {}
    userDict = {}
    cUserDict = {}
    hashtagDict = {}

    fin = open(filename)  
    count = 0
    for doc in fin.readlines():  
        doc = doc.split()
        timeDict[int(doc[0])] = timeDict.setdefault(int(doc[0]), [])
        timeDict[int(doc[0])].append(count)
        userDict[int(doc[2])] = userDict.setdefault(int(doc[2]), [])
        userDict[int(doc[2])].append(count)
        
        cUserDict[int(doc[3])] = cUserDict.setdefault(int(doc[3]), [])
        cUserDict[int(doc[3])].append(count)
        hashtagDict[int(doc[1])] = hashtagDict.setdefault(int(doc[1]), [])
        hashtagDict[int(doc[1])].append(count)
        docs.append([v for v in doc])  
        count += 1
        

    logger.info('hashtag count: %s' % str(len(hashtagDict)))
    logger.info('contact user count: %s' % str(len(cUserDict)))
    logger.info('user count: %s' % str(len(userDict)))
    logger.info('time point count: %s' % str(len(timeDict)))

   
    hdplda = HDPLDA(options.alpha, options.beta, options.gamma, docs, len(id2wordDict))
    
    hdplda_learning(hdplda, options.iteration)
    
    hHDP(hdplda, level=3, num=30, len_id=len(id2wordDict))
    


if __name__ == "__main__":
    main()