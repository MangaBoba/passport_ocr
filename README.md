# passport_ocr
OCR of name, surname and patronymic in passport
_____
You can run demo in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ab6EdorgygbO2vLPgQHRZw9GsEq3_TVq?authuser=0#scrollTo=SfSGzmufOZjQ) (with GPU only)
_____
Train pipeline based on https://github.com/GitYCC/crnn-pytorch

Data for model pre-training https://github.com/wlinna/russian-ocr

Data for model tuning generated with [trgd util](https://github.com/Belval/TextRecognitionDataGenerator) from russian names and surnames [database](https://mydata.biz/ru/catalog/databases/names_db)

For text bounding box detection used EasyOCR default detector.
_____
Limitations of solution:
1. Only aligned data recognition (±5 degrees).
2. Good recognition quality only for upper case letter words.

ToDo: \
- [ ] Name, surname and patronymic among the recognized words.
