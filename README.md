# running instructions 
### base line 
Generate images using the base stable diffusion model without fine tuning. the generated file are stored under base_line_imgs directory.
1. run baseline_sd_sample

### lora fine tuning 
1. add image/text pair to data directory. the default setting is 139 files for 米芾
2. add your wandb api key to the colab 
3. the data_curate directory has more datasets. if desired, copy curated data to the data directory to train on different data set. 
4. run lora_sd_colab using GPU. (CPU won't work) 



# Data 
- effective dataset size: 250
- 褚遂良 24 （have to remove data from 碑, as results are worse.
- 赵孟頫 24
- 文征明 63 
- 米芾 139
- data directory contains english based training data
- data_chinese directory contains chinese based training data. 


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
- ✅ 赵孟頫 add data set
- ✅ 文征明 add data set 
- ✅  Visualize attention layer
- ✅  Visualize conv layer
- [ jieshuh2 ] Adapative lora based on the gradient 

## experiments 
- [ jieshu ] try different rank and alpha
- [ huici ] compare performance of lora on attention only, conv only, and attention+conv.
- [ cancelled ] experiment to see whether we can mix the style.
-  ✅ try different agumentation settings
    - Dataset diversity is more important than aggressive augmentation—cover different calligraphy styles (e.g., 行书, 草书, 隶书).
    - Use higher-resolution crops (e.g., 512x512 or larger) to retain fine-grained stroke details.


## eval!!!!
- notes: Track visual quality with CLIP-based similarity or a calligraphy-style classifier to avoid overfitting.
- [ jieshuh2 ] quantified eval clip
    - base model:
      -  use chinese prompt to generate images, we can't get good images for caligraphy
      -  use english prompt, we can get caligraphy images;
      -  negative prompt
         - use unrelated prompt (e.g. sunset), we can get realistic and diversified images
         - use similar prompt (e.g. text), we can get realistic and diversified images
    - use model tuned with Chinese prompt:
      -  use chinese prompt to generate images, we get images similar to caligraphy but is not readable (CLIP increases)
      -  use english prompt, we can get caligraphy images (CLIP increases);
      -  negative prompt
         - use unrelated prompt (e.g. sunset), we get distorted images baked in with caligraphy style (CLIP decreases)
         - use similar prompt (e.g. text), we can get images similar to caligraphy; (CLIP decreases)
     
- quantitative eval:
    - clip score
    - FID score: compare fake image and real image similarity
    - 






