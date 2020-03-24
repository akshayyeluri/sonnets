import re, string
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation

import nltk
from nltk.corpus import cmudict

####################
# Syllable processing functions
####################

def get_syll_map(words):
    '''
    Get a syllable dict from a set of words using nltk to guess
    '''
    arpabet = cmudict.dict()
    counter = lambda l: sum([any([(d in w) for d in string.digits]) for w in l])
    mapper = {w : list(set(counter(pron) for pron in arpabet[w])) \
                  for w in words if w in arpabet}
    bad_words = [w for w in words if w not in arpabet]
    return mapper, bad_words

def update_syll_map(syll_map, obs_map):
    '''
    Update a syllable map with entries for apostrophied words
    that are in sonnets but in syllable map without
    the apostrophe (e.g. beds', 'th, etc.).
    Also, convert the syll_map to map word_index to syllables instead
    '''
    word_set = set(obs_map.keys())
    too_much = [w for w in word_set if w not in syll_map]
    replace_from = [w.replace("'", '') for w in too_much]
    assert(all([r in syll_map for r in replace_from]))
    syll_map.update({w:syll_map[w2] for w, w2 in zip(too_much, replace_from)})
    missing = [w for w in syll_map if w not in word_set]
    for w in missing:
        del syll_map[w]
    assert(set(syll_map.keys()) == word_set)
    return {obs_map[k]:v for k, v in syll_map.items()}
 
    
    
    
########################################
# Parsing functions for Sonnets
########################################

def parse_text(text, by = 'line', cap='lower', \
               punc_to_drop=re.sub(r"[-']", '', string.punctuation),
               breaks = (0, 4, 8, 12, 14), sonnet_l=14):
    '''
    Parse a text into a sequence of either sonnets, stanzas, or lines.
    Also handle punctuation and capitalization issues
    '''
    # Issues: tokenize as words? Maybe should have some bigrams / word pairs, etc?
    #         just dropping the weird length sonnets, is there better soln?
    
    if punc_to_drop:
        text = text.translate(str.maketrans('', '', punc_to_drop))
    if cap == 'lower':
        text = text.lower()
        
    sonnets = [s[s.find('\n') + 1:] for s in text.split('\n\n\n')]
    # 98 and 125 are NOT 14 line sonnets (15 and 12 resp.)
    sonnets = [s for s in sonnets if len(s.split('\n')) == 14]
    lines = [l.strip() for s in sonnets for l in s.split('\n')]
    
    if by == 'line':
        seqs = lines
    elif by == 'stanza':
        seqs = [' '.join(lines[i+a:i+b]) for i in range(0, len(lines), sonnet_l) \
                                         for a, b in zip(breaks[:-1], breaks[1:])]
    elif by == 'sonnet':
        seqs = [' '.join(lines[i:i+sonnet_l]) for i in range(0, len(lines), sonnet_l)]
        
    return seqs


def rhyme_dict_gen(text, sonnet_l=14, abab_pattern_l=12,
                    connected=False, with_words = False): 
    '''
    Generate a rhyming dictionary.
    '''
    lines = parse_text(text, by='line')                                         
    seqs, obs_map = parse_seqs(lines)
    ind_to_word = obs_map_reverser(obs_map)
                                                                                
    quat = [seqs[i+offset] for i in range(0, len(seqs), sonnet_l)               
                                for offset in range(abab_pattern_l)]            
    coup = [seqs[i+offset] for i in range(0, len(seqs), sonnet_l)               
                                for offset in range(abab_pattern_l, sonnet_l)] 
    pair_ends = [quat[i+offset][-1] for i in range(0, len(quat), 4)  
                                               for offset in [0, 2, 1, 3]]      
    pair_ends += [c[-1] for c in coup]                               
    pairs = list(zip(pair_ends[::2], pair_ends[1::2]))                          
                                                                                
    if connected:
        eq_classes = []
        for (w1, w2) in pairs:
            already_added = False
            for c in eq_classes:
                if w1 in c or w2 in c:
                    c.add(w1); c.add(w2)
                    already_added = True
                    break
            if not already_added:
                c = set(); c.add(w1); c.add(w2)
                eq_classes.append(c)
        d = {w:np.array(list(c)) for c in eq_classes for w in c}
        
    else:
        # Only use rhyming pairings shakespeare specified
        d = {}                                                                      
        for (w1, w2) in pairs:                                                      
            if w1 not in d:                                                         
                d[w1] = []                                                          
            if w2 not in d:                                                         
                d[w2] = []
            d[w1] += [w2] if w2 not in d[w1] else [] 
            d[w2] += [w1] if w1 not in d[w2] else []                             
    
        d = {k:np.unique(v) for k, v in d.items()}
    d_with_words = {ind_to_word[w1] : np.array([ind_to_word[w2] for w2 in v])    
                                      for w1, v in d.items()}                                                                       
    return d_with_words if with_words else d





########################################
# Parsing functions for Limericks
########################################

def parse_lim(text, by = 'line', cap='lower', \
               punc_to_drop=re.sub(r"[-']", '', string.punctuation)):
    '''
    Parse a text into a sequence of either limericks, stanzas, or lines.
    Also handle punctuation and capitalization issues
    '''
    
    if punc_to_drop:
        text = text.translate(str.maketrans('', '', punc_to_drop))
    if cap == 'lower':
        text = text.lower()
        
    limericks = text.split('\n\n')
    # 98 and 125 are NOT 14 line sonnets (15 and 12 resp.)
    limericks = [s for s in limericks if len(s.split('\n')) == 5]
    lines = [l.strip() for s in limericks for l in s.split('\n')]
    
    if by == 'line':
        seqs = lines
    elif by == 'limerick':
        seqs = [' '.join(lines[i:i+5]) for i in range(0, len(lines), 5)]
        
    return seqs


def rhyme_dict_lim(text, sonnet_l=14, abab_pattern_l=12,
                    connected=False, with_words = False): 
    '''
    Generate a rhyming dictionary for limericks.
    '''
    lines = parse_lim(text, by='line')                                         
    seqs, obs_map = parse_seqs(lines)
    ind_to_word = obs_map_reverser(obs_map)
                                                                                
    main = [seqs[i+offset][-1] for i in range(0, len(seqs), 5)               
                                for offset in [0, 1, 4]]            
    coup = [seqs[i+offset][-1] for i in range(0, len(seqs), 5)               
                                for offset in [2, 3]]
    


    pair_ends = list(np.ravel([[(a,b),(b,c),(c,a)] for (a,b,c) in np.array(main).reshape(-1, 3)]))
    pair_ends += coup                               
    pairs = list(zip(pair_ends[::2], pair_ends[1::2]))                          
                                                                                
    if connected:
        eq_classes = []
        for (w1, w2) in pairs:
            already_added = False
            for c in eq_classes:
                if w1 in c or w2 in c:
                    c.add(w1); c.add(w2)
                    already_added = True
                    break
            if not already_added:
                c = set(); c.add(w1); c.add(w2)
                eq_classes.append(c)
        d = {w:np.array(list(c)) for c in eq_classes for w in c}
        
    else:
        # Only use rhyming pairings shakespeare specified
        d = {}                                                                      
        for (w1, w2) in pairs:                                                      
            if w1 not in d:                                                         
                d[w1] = []                                                          
            if w2 not in d:                                                         
                d[w2] = []
            d[w1] += [w2] if w2 not in d[w1] else [] 
            d[w2] += [w1] if w1 not in d[w2] else []                             
    
        d = {k:np.unique(v) for k, v in d.items()}
    d_with_words = {ind_to_word[w1] : np.array([ind_to_word[w2] for w2 in v])    
                                      for w1, v in d.items()}                                                                       
    return d_with_words if with_words else d





####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True,
                        M = 100000):
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission(M)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################

def parse_seqs(seqs):
    obs_counter = 0
    obs = []
    obs_map = {}
    
    for seq in seqs:
        obs_elem = []
        for word in seq.split():
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        # Add the encoded sequence.
        obs.append(obs_elem)
    
    return obs, obs_map


def obs_map_reverser(obs_map):
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return ' '.join(sentence).capitalize() + '...'


####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()


def get_stats(hmm, obs_map, M = 100000):
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []
    cmu_dict = cmudict.dict()
    # Generate a large emission.
    emission, states = hmm.generate_emission(M)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)


    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)
        tokenized = nltk.word_tokenize(sentence_str)
        tagged = nltk.pos_tag(tokenized, tagset = 'universal')
        tags = np.ndarray.tolist(np.array(tagged)[:,1])
        print("State %d"% i)
        print("Nouns: %f%%"% (tags.count('NOUN')/len(sentence) * 100))
        print("Verbs: %f%%"% (tags.count('VERB')/len(sentence) * 100))
        print("Pronouns: %f%%"% (tags.count('PRON')/len(sentence) * 100))
        print("Adjectives: %f%%"% (tags.count('ADJ')/len(sentence) * 100))
        print("Adverbs: %f%%"% (tags.count('ADV')/len(sentence) * 100))
        print("Adpositions: %f%%"% (tags.count('ADP')/len(sentence) * 100))
        print("Conjunctions: %f%%"% (tags.count('CONJ')/len(sentence) * 100))
        print("Determiners: %f%%"% (tags.count('DET')/len(sentence) * 100))
        print("Cardinal Numbers: %f%%"% (tags.count('NUM')/len(sentence) * 100))
        print("Particles: %f%%"% (tags.count('PRT')/len(sentence) * 100))
        print("Other: %f%%"% (tags.count('X')/len(sentence) * 100))
        print("Punctuation: %f%%"% (tags.count('.')/len(sentence) * 100))
        print()

        syll_dict = {}
        stress_dict = {}
        for word in sentence:
            if word in cmu_dict:
                pron = cmu_dict[word][0]
                stress = ""
                for j in pron:
                    if j[-1] == '0':
                        stress += 'U'
                    elif j[-1] == '1' or j[-1] == '2':
                        stress += 'S'
                if stress in stress_dict:
                    stress_dict[stress] += 1
                    syll_dict[len(stress)] += 1
                else:
                    stress_dict[stress] = 1
                    syll_dict[len(stress)] = 1

        print(syll_dict)
        print(stress_dict)
        print()
    
    
####################
# HMM ANIMATION FUNCTIONS
####################

def animate_emission(hmm, obs_map, M=8, height=12, width=12, delay=1):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06
    
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, '', fontsize=24)
        
    # Make the arrows.
    zorder_mult = n_states ** 2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)
            
            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(x_i + (r/d + arrow_p1) * dx + arrow_p2 * dy,
                                 y_i + (r/d + arrow_p1) * dy + arrow_p2 * dx,
                                 (1 - 2 * r/d - arrow_p3) * dx,
                                 (1 - 2 * r/d - arrow_p3) * dy,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))
            else:
                arrow = ax.arrow(x_i, y_i, 0, 0,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color('red')
            elif i == 1:
                arrows[states[0]][states[0]].set_color((1 - hmm.A[states[0]][states[0]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')
            else:
                arrows[states[i - 2]][states[i - 1]].set_color((1 - hmm.A[states[i - 2]][states[i - 1]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')

            # Set text.
            text.set_text(' '.join([obs_map_r[e] for e in emission][:i+1]).capitalize())

            return arrows + [text]

    # Animate!
    print('\nAnimating...')
    anim = FuncAnimation(fig, animate, frames=M+delay, interval=1000)

    return anim

    # honestly this function is so jank but who even fuckin cares
    # i don't even remember how or why i wrote this mess
    # no one's gonna read this
    # hey if you see this tho hmu on fb let's be friends



