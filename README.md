# Sonnets
Shakespearean sonnet generation using HMMs (Hidden Markov Models) and RNNs (Recurrent Neural Networks)

# Files
1. HMM.py: Implementation of HMMs, also has implementations of subclasses ShakespeareHMM and Lim_HMM (for sonnets
           and limericks in particular), where the subclass instances have a "generate_sonnet" or "generate_limerick" method
           using the underlying HMM to generate a particular type of poetry
2. helper.py: Variety of helper functions for text preprocessing and animation / visualization. Useful functions include 
    1. parse_text / parse_lim -- parse a corpus by line, stanza, or poem (use parse_lim for limericks)
    2. get_syll_map / update_syll_map -- functions for getting / correcting a syllable dictionary mapping words to
                                   a list of possible numbers of syllables in said words
    3. rhyme_dict_gen / rhyme_dict_lim -- generate a dictionary mapping words to rhyming words from a body of text
    4. get_stats -- get statistics for the states in an HMM, useful for visualization / HMM analysis.
3. hmm.ipynb: Notebook demonstrating preprocessing sonnets, training an hmm with sonnet data, pickling (loading/saving) a 
           trained model, creating a ShakespeareHMM from a trained hmm and using said Shakespeare HMM to generate a sonnet
4. viz_hmm.ipynb: Notebook for hmm visualization / analysis
5. lim_hmm.ipynb: Notebook demonstrating preprocessing limericks, training an hmm with limerick data, pickling (loading/saving) a 
           trained model, creating a Lim_HMM from a trained hmm and using said Lim_HMM to generate a limerick
6. RNN.ipynb: Notebook demonstrating RNN training / poem generation with RNNs
7. rnn_torch.ipynb: same as RNN.ipynb, but implemented in pytorch


# Examples of Model generated poems 
## HMM Sonnet
=================================

Still touch music me jewel may nor sing,  
And now see better outward breath all east,  
Love through verse truth hand seeming feeling wing,  
Compounds worms me be their and 'i in west,  
You usest extreme like then i will straight,  
Of which love mother's to prognosticate,  
That yet thy could thou a eye something bait,  
Breathers where thou this are of leases date,  
O say with thou when can disabled,  
Be art acknowledge i oblivion near,  
Your thanks bars each have grace his strumpeted,  
Confound unrespected supposed her cheer,  
Shall note beauty well of gluttoning love,  
Shapes strive wish i self art do wilt were move,  

## HMM Limerick
=================================

And he of off his a fart,  
At who found whose supplied art,  
A was launch wife lewd,  
For the looky strewed,  
Minuscule his and god part,  

## RNN Sonnet
=================================

Still touch music me jewel may nor sing,  
And now see better outward breath all east,  
Love through verse truth hand seeming feeling wing,  
Compounds worms me be their and 'i in west,  
You usest extreme like then i will straight,  
Of which love mother's to prognosticate,  
That yet thy could thou a eye something bait,  
Breathers where thou this are of leases date,  
O say with thou when can disabled,  
Be art acknowledge i oblivion near,  
Your thanks bars each have grace his strumpeted,  
Confound unrespected supposed her cheer,  
Shall note beauty well of gluttoning love,  
Shapes strive wish i self art do wilt were move,  


          
         
        