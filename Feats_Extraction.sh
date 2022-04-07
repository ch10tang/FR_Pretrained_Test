# Extract in dictionary
# D:/Josh2/FR_Pretrained_Test
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_ArcFace_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=IR-50 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/ArcFace_Aligned_112_112 -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_LightCNN_128_128 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=Light_CNN_29 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/LightCNN_Aligned_128_128 -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_VGGFace2_224_224 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_Aligned_224_224 -batch-size=8
#
# python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeF_Sub4979_Img74132_0_30_WarpAffine7refs -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeS_Sub4979_Img74861_30_60_WarpAffine7refs -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeS_Sub4979_Img74861_30_60_WarpAffine7refs -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeP_Sub4979_Img74522_60_92_WarpAffine7refs_Flip -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeP_Sub4979_Img74522_60_92_WarpAffine7refs_Flip -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeF_Sub4979_Img74596_0_60_balance_WarpAffine7refs -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeP_Sub4979_Img77566_30_92_WarpAffine7refs -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeP_Sub4979_Img77566_30_92_WarpAffine7refs -batch-size=8
python Main_HashTable.py -data-place=F:/Database/IJBC/IJBC_Official_WarpAffine7refs_112_112 -csv-file=../DataList/IJBC_Official_Aligned.csv -model-select=VGGFace2_TypeA_Sub4979_Img74060_0_30_60_90_balance_WarpAffine7refs -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_TypeA_Sub4979_Img74060_0_30_60_90_balance_WarpAffine7refs -batch-size=8


# Extract in np.array (InsightFace)
# D:/Josh2/FR_Pretrained_Test
python Main_InsightFace.py -data-place=F:/Database/IJBC/IJB-C_InsightFace_MagFace_112_112 -csv-file=../DataList/IJBC_InsightFace_ArcFace.csv -model-select=IR-50 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/ArcFace_Aligned_112_112_InsightFace -batch-size=8
python Main_InsightFace.py -data-place=F:/Database/IJBC/IJB-C_InsightFace_LightCNN_128_128 -csv-file=../DataList/IJBC_InsightFace.csv -model-select=Light_CNN_29 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/LightCNN_Aligned_128_128_InsightFace -batch-size=8
python Main_InsightFace.py -data-place=F:/Database/IJBC/IJB-C_InsightFace_VGGFace2_224_224 -csv-file=../DataList/IJBC_InsightFace.csv -model-select=VGGFace2 -generate-place=../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_Aligned_224_224_InsightFace -batch-size=8


