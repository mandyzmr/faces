#!/bin/bash

# /bin/bash create_masked_dataset.sh > log.txt 2>&1 & 
python preprocess/create_masked_dataset.py --start_p=0.95 --end_p=1.0 --unmasked=True --masked=True --mask=True
									  