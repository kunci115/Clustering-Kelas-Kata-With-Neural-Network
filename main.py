#buat run harus install pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class StemmerGenerator():
  def __init__(self):
    self.stemmer = StemmerFactory().create_stemmer()
    self.readMe = open('kbbi.csv','r').readlines()
    self.list_kata_dasar = open('kata_dasar.txt','w')
    self.list_kata_imbuh = open('kata_imbuh.txt','w')

  def stemm(self):
    for i in self.readMe:
        unstammed =i.strip()
        stemmed   = self.stemmer.stem(i)
        if str(stemmed) == str(unstammed):
            self.list_kata_dasar.write(stemmed+'\n')
        elif str(stemmed) != str(unstammed):
            self.list_kata_imbuh.write(unstammed+'\n')
    return (self.list_kata_dasar, self.list_kata_imbuh)

s = StemmerGenerator()
kd, ki = s.stemm()
print('ini list katadasar',kd)
print('ini list kata imbuh',ki)
#
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# readMe = open('kbbi.csv','r').readlines()
# list_kata_dasar = []
# list_kata_imbuh = []
# # stemming process
# def stemm():
#     for i in readMe:
#         unstammed =i.strip()
#         stemmed   = stemmer.stem(i)
#
#
#         if str(stemmed) == str(unstammed):
#             list_kata_dasar.append(stemmed)
#
#         elif str(stemmed) != str(unstammed):
#             list_kata_imbuh.append(unstammed)
# stemm()
# print('ini list ktadasar',list_kata_dasar)
# print('ini list kata imbuh',list_kata_imbuh)



#
# #buat run harus install pip install Sastrawi
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#
# # create stemmer
# class stemmer():
#     def __init__(self):
#         self.factory = StemmerFactory()
#         self.stemmer = self.factory.create_stemmer()
#         self.readMe = open('kbbi.csv','r').readlines()
#         self.list_kata_dasar = []
#         self.list_kata_imbuh = []
#
#     def stemm(self):
#         for i in self.readMe:
#             unstammed =i.strip()
#             stemmed   = stemmer.stem(i)
#
#
#             if str(stemmed) == str(unstammed):
#                 list_kata_dasar.append(stemmed)
#
#             elif str(stemmed) != str(unstammed):
#                 list_kata_imbuh.append(unstammed)
# s = stemmer()
# s.stemm()
# for kd in s.list_kata_dasar:
#  print(kd)