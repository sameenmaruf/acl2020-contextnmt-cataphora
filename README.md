This repository contains the code (a DyNet implementation of Tranformer-HAN-encoder (https://www.aclweb.org/anthology/D18-1325/)) used in our paper, accepted for publication at ACL 2020:

KayYen Wong, Sameen Maruf and Gholamreza Haffari. Contextual Neural Machine Translation Improves Translation of Cataphoric Pronouns. 

The data can be downloaded from: https://drive.google.com/file/d/117Yb8bW1zJi9dNwcBT5mf7Hikda6fybY/view?usp=sharing 

Please cite our paper if you use the data or the code. 

*Note*:

Whatever data set you use with our implementation, you need to add the BOS and EOS tokens (represented by `<s>` and `</s>`) to the source and target sentences yourself.

Also you may need to unkify (replace by `<unk>`) the tokens in dev and test sets which do not exist in the training set.

### Contacts

Cataphora Test Suite: KayYen Wong (kayyenwong@gmail.com)

Code: Sameen Maruf (sameen.maruf@monash.edu)
