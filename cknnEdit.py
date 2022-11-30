from _operator import itemgetter
from math import sqrt
import random
import time
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from itertools import repeat
from functools import partial
import parmap
class ContextKNN:
    '''
    ContextKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, data, k=100, sample_size=500, sampling='recent',  similarity = 'cosine', remind=False, pop_boost=0, extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        
        self.sim_time = 0
        
        self.data = data
        self.fit(self.data)
        self.multi_proc_res_list = Manager().list()
        self.multi_proc_scr_list = Manager().list()
    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        index_session = train.columns.get_loc( self.session_key )#col position
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        
        session = -1
        session_items = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            #row = 1 row tuple data
            # cache items of sessions
            
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        #self.item_session_map item : appeared session list
        #self.session_item_map session : appeared item list
        
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, skip=False):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
#         gc.collect()
#         process = psutil.Process(os.getpid())
#         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')
        
        if( self.session != session_id ): #new session
            
            if( self.extend ):#default False
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
                
            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()
        
        self.session_items.append( input_item_id )# now test session

        if skip:
            return

        neighbors = self.find_neighbors( set(self.session_items), input_item_id, session_id )
        scores = self.score_items( neighbors )
        ret = []
        ret.append(scores)
        # add some reminders
        if self.remind:
             
            reminderScore = 5
            takeLastN = 3
             
            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                #reminderScore = reminderScore + (cnt/100)
                 
                oldScore = scores.get( elem )
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                #print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt/100)
                 
                scores.update({elem : newScore})
        
        #push popular ones
        if self.pop_boost > 0:
               
            pop = self.item_pop( neighbors )
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})
         
        
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        
        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        
        if self.normalize:
            series = series / series.max()
        
        return series, ret

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session( session )
            for item in items:
                
                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})
                    
                if( result.get(item) > max_pop ):
                    max_pop =  result.get(item)
         
        for key in result:
            result.update({key: ( result[key] / max_pop )})
                   
        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / (sqrt(la) * sqrt(lb))

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id )
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                
            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
                        

    def calc_similarity(self, session_items, sessions ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session( session )
            

            similarity = getattr(self , self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))
                
        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors )
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            
            for item in items:
                old_score = scores.get( item )
                new_score = session[1]
                
                if old_score is None:
                    scores.update({item : new_score})
                else: 
                    new_score = old_score + new_score
                    scores.update({item : new_score})
                    
        return scores
    
    def predict(self, ses_id, it_id, its_id):
        #print( ses_id, it_id, its_id)
        #print('here1111')
        ret, scr = self.predict_next(ses_id, it_id, its_id)
        Sorted_ret = ret.sort_values(ascending=False)
        Sorted_ret_label = Sorted_ret.index.values
        return Sorted_ret_label[:20], scr
    
    def predict_tr(self, ts_data):
        predicted_items = []
        targets = np.zeros(shape=1)
        its_id = np.unique(self.data.loc[:,'item_id'].values).astype(np.int64)# all items id
        scr_r = []
        for data_session, ins in tqdm(enumerate(ts_data.inputs), total=len(ts_data.inputs)):
            data_session += 1
            ins = np.delete(ins, np.where(ins == 0))
            for item in ins:
                if item in its_id:
                    items, scr = self.predict(data_session, item, its_id)
                else:
                    items = np.zeros(shape=20)
                    scr = []
                    break
            #items: predicted 20 item_ids
            #scr: items score
            scr_r.append(scr)
            predicted_items.append(items)

        predicted_items = np.asarray(predicted_items)
        predicted_items = np.reshape(predicted_items, (-1, 20))

        targets = ts_data.targets - 1
        
        hit, mrr = [], []
        for predicted_item, target in tqdm(zip(predicted_items, targets), total=len(predicted_items)):
            hit.append(np.isin(target, predicted_item))
            if len(np.where(predicted_item == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(predicted_item == target)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        
        return mrr, hit, scr_r
    
    def predict_ts(self, ts_data_list):
        index = ts_data_list[1]
        ts_data_list = ts_data_list[0]
        its_ids = np.unique(self.data.loc[:,'item_id'].values).astype(np.int64)# all items id
        predicted_items = []
        scr_r = []
        
        for data_session, ins in tqdm(enumerate(ts_data_list), total=len(ts_data_list), position=index):
            data_session += 1
            ins = np.delete(ins, np.where(ins == 0))
            for item in ins:
                if item in its_ids:
                    items, scr = self.predict(data_session, item, its_ids)
                else:
                    items = np.zeros(shape=20)
                    scr = []
                    break
            #items: predicted 20 item_ids
            #scr: items score
            
            scr_r.append(scr)
            predicted_items.append(items)
        predicted_items = np.asarray(predicted_items)
        predicted_items = np.reshape(predicted_items, (-1, 20))
        self.multi_proc_scr_list[index] = scr_r
        self.multi_proc_res_list[index] = predicted_items

        return None
    
    def split_data(self, ts_data, n_proc):
        ts_data_list = []
        total_index = ts_data.inputs.shape[0]
        start_index = 0
        for calc in range(n_proc):
            self.multi_proc_res_list.append(0)
            self.multi_proc_scr_list.append(0)
            calc+=1
            index = int(total_index*calc/n_proc)
            data = (ts_data.inputs[start_index:index,:], calc - 1)
            ts_data_list.append(data)
            start_index = index
        return ts_data_list
    
    def predicted_items_calc(self):
        predicted_items = np.array(self.multi_proc_res_list[0])
        if len(self.multi_proc_res_list) > 1:
            etc = self.multi_proc_res_list[1:]
            for pls in etc:
                predicted_items = np.concatenate((predicted_items, pls), axis=0)
        return predicted_items
    
    def predicted_scr_sum(self):
        scr_sum = self.multi_proc_scr_list[0]
        if len(self.multi_proc_scr_list) > 1:
            etc = self.multi_proc_scr_list[1:]
            for seqs in etc:
                for seq in seqs:
                    scr_sum.append(seq)
        return scr_sum
    
    def multi_predict_ts_data(self, ts_data, n_proc):
        self.multi_proc_scr_list = Manager().list()
        self.multi_proc_res_list = Manager().list()
        #reset
        ts_data_list = self.split_data(ts_data, n_proc)
        
        pool = Pool(n_proc)
        pool.map(self.predict_ts, tqdm(ts_data_list))
        pool.close()
        pool.join()
        #parmap.map(self.predict_ts, ts_data_list, pm_pbar=True, pm_processes=n_proc )
        predicted_items = self.predicted_items_calc()
        
        targets = ts_data.targets - 1
        hit, mrr = [], []
        for predicted_item, target in tqdm(zip(predicted_items, targets), total=len(predicted_items)):
            #print(predicted_item, target)
            #input()
            hit.append(np.isin(target, predicted_item))
            if len(np.where(predicted_item == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(predicted_item == target)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        scr_r = self.predicted_scr_sum()
        return mrr, hit, scr_r