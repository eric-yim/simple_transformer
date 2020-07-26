### Transformer as replacement to RNN
It is often the goal to simply predict the next step in a sequence, or the next word in a sentence. Though basic code for language-to-language transformers is easily found, I've not been able to find code for a simplified transformer than can just predict the next step. It is hard to edit code within the transformers because understanding the inner workings of the transformer is quite involved.

For my friends who don't want to think about how a transformer works, but want to use its power, here is a set up you can use.

An example input is a matrix (Batch x SequenceLength) composed of integers. The output is a matrix (Batch x 1), which is the next term in the sequence.

The transformer network has not been altered from its origin.

The training .py has been set up for the inputs/outputs as described. 

### Train on an Arbitrary Pattern

As a dummy test, run the following.
```
python3 singleout_train.py
```
It trains on a dummy sequence with a set pattern.

For example,

[1,2,4,3,5,6,7,9,8,1,2,4,3,...]

The transformer should then be able to predict what comes next, one at a time.

[5,6,7,9,8,1,2 ...]


Model parameters are in the singleout_config.py

### Make Sure Its Working
```
python3 singleout_eval.py
```
You should be able to get the following to print out.

Input Prompt: [4]

Completed Prompt: [[4 3 5 6 7 9 8 1]]


