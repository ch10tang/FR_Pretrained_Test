#!/usr/bin/env bash

OUTPUT=./test/
FF=MS1M_FF_Sub19969_Img94819_0_30
SS=MS1M_SS_Sub10837_Img48367_30_60
PP=MS1M_PP_Sub1532_Img5995_60_90_over3
FS=MS1M_FS_Sub20000_Img94835_0_60
SP=MS1M_SP_Sub10356_Img71855_30_90
FP=MS1M_FP_Sub19971_Img94446_0_30_60_90
DATA_PLACE=/media/tang/Liang2/Database/MS1M
MODEL_PATH=/media/tang/Josh22/06_Model/2021-2022/POE_face.evoLVe.PyTorch-master/MS1M
DATA_LIST=./DataList


specific_model=${MODEL_PATH}/FF/${FF}_LR001
epoch=9
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv

specific_model=${MODEL_PATH}/SS/${SS}_LR001
epoch=27
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv

specific_model=${MODEL_PATH}/PP/${PP}_LR001
epoch=62
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv

specific_model=${MODEL_PATH}/FS/${FS}_LR001
epoch=24
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv

specific_model=${MODEL_PATH}/SP/${SP}_LR001
epoch=9
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv

specific_model=${MODEL_PATH}/FP/${FP}_LR001
epoch=42
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FF} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FF}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${PP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${PP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FS} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FS}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${SP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${SP}.csv
python Main_HashTable_MS1M.py -data-place=${DATA_PLACE}/${FP} -model-path=${specific_model} -epoch=${epoch} -csv-file=${DATA_LIST}/${FP}.csv



