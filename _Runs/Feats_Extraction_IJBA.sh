# Extract in dictionary
# D:/Josh2/FR_Pretrained_Test

# SOTAs
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_ArcFace_112_112_22-05-03New \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=IR-50 \
			  -generate-place=../IJBA_Tutorial/_Features/ArcFace_Aligned_112_112 \
			  -batch-size=8
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_LightCNN_128_118_22-05-03New \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=Light_CNN_29 \
			  -generate-place=../IJBA_Tutorial/_Features/LightCNN_Aligned_128_128 \
			  -batch-size=8
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_VGGFace2_184_224_22-05-03New \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=VGGFace2 \
			  -generate-place=../IJBA_Tutorial/_Features/IJBA_VGGFace2_184_224 \
			  -batch-size=8

# Frontal
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_POEs_112_112_22-05-03New \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs \
			  -generate-place=../IJBA_Tutorial/_Features/VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs \
			  -batch-size=8
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_POEs_112_112_22-05-03New \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs \
			  -generate-place=../IJBA_Tutorial/_Features/VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs \
			  -batch-size=8
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_POEs_112_112_22-05-03New_Flip \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs_Flip \
			  -generate-place=../IJBA_Tutorial/_Features/VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs_Flip \
			  -batch-size=8
python Main_HashTable.py -data-place=../Database/IJBA/IJBA_POEs_112_112_22-05-03New_Flip \
			  -csv-file=./DataList/IJBA_WarpAffine7refs_112_112_22-05-03New.csv \
			  -model-select=VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs_Flip \
			  -generate-place=../IJBA_Tutorial/_Features/VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs_Flip \
			  -batch-size=8




