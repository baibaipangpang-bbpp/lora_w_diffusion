# running instructions 
### base line 
1. run baseline_sd_sample. the generated file are stored under base_line_imgs directory.

### lora fine tuning 
1. add image/text pair to data directory. the default setting is 139 files for 米芾
2. add your wandb api key to the colab 
3. the data_curate directory has more datasets. if desired, copy curated data to the data directory to train on different data set. 
4. run lora_sd_colab using GPU. (CPU won't work) 



# TODO
## communication with TA
- [@jieshuh2 ] need to sync up with TA for communication and alignment purpose.


## Future work idea: 
* The limitation of max_length imposed by CLIP is 77. This is a significant limitation for the task on hand, as the caligraphy is usually a piece of text, I was hoping to get the model to understand the 1:1 mapping of the words.  Encorporating T5/BERT model to text embedding could be explored to eleviate the limitations. 



## Things to do
- ✅ Wandb integration 
- ✅ implement lora to conv layer  
- ✅ Add “书法” to all text.
- ✅ data augmentation
- ✅ add data agumentation w/ kornia
- ✅ 褚遂良 add data set
- [ @huici ] Visualize attention layer
- [ @huici ] Visualize conv layer
- [ jieshuh2 ] Adapative lora based on the gradient 
- [cancel] 赵孟頫 add data set
- [cancel] 颜真卿 data set to


## experiments 
- [ zhiyin ] try different dropout
- [ zhiyin ] compare performance of lora on attention only, conv only, and attention+conv.
- [ zhiyin ] experiment to see whether we can mix the style.
- [ zhiyin ] try different agumentation settings


## eval!!!!
- [ jieshuh2 ] quantified eval clip 





