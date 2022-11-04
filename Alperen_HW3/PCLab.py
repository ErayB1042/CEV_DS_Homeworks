class PCLab():
    def __init__(self, sınıfadı="VB101", bilgisayar_sayısı=10,labdoluluk_durumu="Boş",öğrenci_sayısı=10,ogretmen=["Ahmet"]):
        self.sınıfadı = sınıfadı
        self.bilgisayar_sayısı = bilgisayar_sayısı
        self.öğrenci_sayısı = öğrenci_sayısı
        self.labdoluluk_durumu = labdoluluk_durumu
        self.ogretmen=ogretmen

    def __len__(self):
        return len(self.ogretmen)

    def bilgisayar_ekle(self):

        while True:
            cevap = input(f"Labdaki Mevcut bilgisayar sayısı : {self.bilgisayar_sayısı}, bilgisayar sayısını 1 arttırmak 1 için '>'\nBilgisayar sayısını azaltmak için '<'\nÇıkış için 'q'\n")

            if cevap == ">":

                if self.bilgisayar_sayısı != 25:
                    self.bilgisayar_sayısı += 1
                    print("Bilgisayar Sayısı: {}".format(self.bilgisayar_sayısı))
                elif self.bilgisayar_sayısı ==25:
                    print("Ne yazıkki Lab'ın büyüklüğü daha fazla bilgisayar alımına izin vermemekte.Lab'a Maksimum 25 Bilgisayar sığabilmekte ve maksimum bilgisayar sayısına ulaşılmış bulunmakta!!!")

            elif cevap == "<":
                if self.bilgisayar_sayısı != 0:
                    self.bilgisayar_sayısı -= 1
                    print("Bilgisayar Sayısı: {}".format(self.bilgisayar_sayısı))

                elif self.bilgisayar_sayısı == 0:
                    print("Labdaki mevcut bilgisayar Sayısı 0'dır!!!\n Daha fazla azaltma yapılamaz!")

            else:
                print("Mevcut Bilgisayar Sayısı: {}".format(self.bilgisayar_sayısı))
                break

    def lab_doluluk_durumu(self):

        if (self.öğrenci_sayısı > 2*self.bilgisayar_sayısı) or (self.öğrenci_sayısı == 2*self.bilgisayar_sayısı):
            print("Labdaki Doluluk Durumu: Dolu , Labdaki Bilgisayarların tamamı kullanımda!")
            self.labdoluluk_durumu="Dolu"

        else:
            print("Labdaki Doluluk Durumu: Boş , Labda boşta/kullanıma hazır bilgisayar bulunmakta...")
            self.labdoluluk_durumu="Boş"

    def öğrenci_ekle(self):

        while True:
            cevap = input("Labdaki Mevcut Öğrenci sayısı: {}, öğrenci sayısını 1 arttırmak için '>'\nÖğrenci sayısını azaltmak için '<'\nÇıkış için 'q'\n".format(self.öğrenci_sayısı))

            if cevap == ">":

                if self.öğrenci_sayısı != 0:
                    self.öğrenci_sayısı += 1
                    print("Öğrenci Sayısı:", self.öğrenci_sayısı)

            elif cevap == "<":

                if self.öğrenci_sayısı != 50:
                    self.öğrenci_sayısı -= 1
                    print("Öğrenci Sayısı:", self.öğrenci_sayısı)
                else:
                    print("Daha fazla öğrenci eklenememekte.")
            else:
                print("Labdaki Mevcut Öğrenci sayısı: {}".format(self.öğrenci_sayısı))
                break

    def ogretmen_ekle(self):
        ogretmenadı=input("Eklemek istediğiniz öğretmen/-lerin adını giriniz...")
        print("Öğretmen Sınıfa görevlendiriliyor...")
        self.ogretmen.append(ogretmenadı)
        print("Öğretmen listesi: {}".format(self.ogretmen))

    def __str__(self):
        return "Sınıf Kodu: {}\nBilgisayar Sayısı: {}\nÖğrenci Sayısı: {}\nDoluluk Durumu: {}\nÖğretmen/-ler: {}".format(self.sınıfadı, self.bilgisayar_sayısı, self.öğrenci_sayısı, self.labdoluluk_durumu,self.ogretmen)
6
lab=PCLab()

print("""
    1. Bilgisayar Ekle
    2. Lab Doluluk oranına bak.
    3. Öğrenci Ekle
    4. Öğretmen Sayısı
    5. Tüm Bilgileri Getir.
    6. Öğretmen Ekle.
    7. Çıkış
""")

while True:
    cevap = input("Seçim Yapınız: ")
    if cevap == "1":
        lab.bilgisayar_ekle()
    elif cevap == "2":
        lab.lab_doluluk_durumu()
    elif cevap == "3":
        lab.öğrenci_ekle()
    elif cevap=="4":
        print("Öğretmen sayısı: ",len(lab))
    elif cevap=="5":
        print(lab)
    elif cevap=="6":
        lab.ogretmen_ekle()

    else:
        print("Çıkış Yapılıyor...")
        break