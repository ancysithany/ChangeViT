ChangeViT
Paper for ***[ChangeViT: Unleashing Plain Vision Transformers for Change Detection ](https://arxiv.org/pdf/2406.12847).***

[Duowang Zhu](https://scholar.google.com/citations?user=9qk9xhoAAAAJ&hl=en&oi=ao), [Xiaohu Huang](https://scholar.google.com/citations?user=sBjFwuQAAAAJ&hl=en&oi=ao), Haiyan Huang, Zhenfeng Shao, Qimin Cheng

[[paper]](https://arxiv.org/pdf/2406.12847)

##TRANSFERLEARNING
  
##DATASET   
       
- Download the [LEVIR-CD](https://chenhao.in/LEVIR/)

- Crop each image in the dataset into 256x256 patches.

- Prepare the dataset into the following structure and set its path in the [config]. LEVIR is already in the following structure
    ```
    ├─Train
        ├─A          jpg/png
        ├─B          jpg/png
        └─label      jpg/png
    ├─Val
        ├─A 
        ├─B
        └─label
    ├─Test
        ├─A
        ├─B
        └─label
    ```

### Checkpoint
- Download the pre-weights [ViT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth), and [ViT-S](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth), then put them into checkpoints folder.

## Dependency
```
pip install -r requirements.txt
```

## Training
```
python main.py --file_root LEVIR --max_steps 80000 --model_type small --batch_size 16 --lr 2e-4 --gpu_id 0
```

## Inference
```
python eval.py --file_root LEVIR --max_steps 80000 --model_type small --batch_size 16 --lr 2e-4 --gpu_id 0
```

## License
ChangeViT is released under the [CC BY-NC-SA 4.0 license](LICENSE).


## Acknowledgement
This repository is built upon [DINOv2](https://github.com/facebookresearch/dinov2) and [A2Net](https://github.com/guanyuezhen/A2Net). Thanks for those well-organized codebases.


## Citation
```bibtex
@article{zhu2024changevit,
  title={ChangeViT: Unleashing Plain Vision Transformers for Change Detection},
  author={Zhu, Duowang and Huang, Xiaohu and Huang, Haiyan and Shao, Zhenfeng and Cheng, Qimin},
  journal={arXiv preprint arXiv:2406.12847},
  year={2024}
}
```
