## ContraNovo: A Contrastive Learning Approach to Enhance De Novo Peptide Sequencing

##### Abstract

De novo peptide sequencing from mass spectrometry (MS) data is a critical task in proteomics research. Traditional de novo algorithms have encountered a bottleneck in accuracy due to the inherent complexity of proteomics data. While deep learning-based methods have shown progress, they reduce the problem to a translation task, potentially overlooking critical nuances between spectra and peptides. In our research, we present ContraNovo, a pioneering algorithm that leverages contrastive learning to extract the relationship between spectra and peptides and incorporates the mass information into peptide decoding, aiming to address these intricacies more efficiently. Through rigorous evaluations on two benchmark datasets, ContraNovo consistently outshines contemporary state-of-the-art solutions, underscoring its promising potential in enhancing de novo peptide sequencing.



#### Reproduce Steps

- To get resource and create the conda environment.

  ```
  git clone git@github.com:BEAM-Labs/ContraNovo.git
  cd ContraNovo
  conda env create -f environment.yml
  ```

- To install the torch 1.13.0.

  ```
  #Example:
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
  ```

- Run ContraNovo test on bacillus.10k.mgf

  ```
  python -m ContraNovo.ContraNovo  --mode=eval --peak_path=./ContraNovo/bacillus.10k.mgf --model=./ContraNovo/ContraNovo.ckpt
  ```

  

