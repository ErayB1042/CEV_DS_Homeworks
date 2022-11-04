import time
import random



class Kumanda():

    def __init__(self, tv_durum="Kapalı",tv_ses=0,kanal_listesi=["TRT"],kanal="TRT",ınternet="Bağlı Değil!",youtube="Kapalı",ucboyut_mode="Kapalı",yazılım_guncelle= "Kapalı"):

        self.tv_durum=tv_durum
        self.tv_ses=tv_ses
        self.kanal_listesi=kanal_listesi
        self.kanal= kanal
        self.ınternet=ınternet
        self.youtube=youtube
        self.ucboyut_mode=ucboyut_mode
        self.yazılım_guncelle =yazılım_guncelle

    def tv_ac(self):

        if (self.tv_durum  == "Kapalı"):
            print("Tv açılıyor")
            self.tv_durum="Açık"
        else:
            print("Tv zaten acik")

    def tv_kapat(self):
        
        if (self.tv_durum  == "Kapalı"):
            print("Tv zaten kapalı")
        
        else:
            print("Tv zaten kapaniyor")
            self.tv_durum="Kapalı"
    

    def ses_ayari(self):

        while True:

            cevap=input("Sesi artır :>\nSesi azalt: <\nÇıkış: q")

            if (cevap=='>'):
                if self.tv_ses>=100:
                    print("Max Ses")

                self.tv_ses+=1
                print("Tv Sesi: {}".format(self.tv_ses))
            
            elif (cevap=="<"):
                if (self.tv_ses<=0):
                    print("Min ses")

                self.tv_ses-=1
                print("tv_ses: {}",format(self.tv_ses))
            elif (cevap=="q"):
                break
            else:
                print("hatalı tuşlama")

    def kanal_ekle(self,kanal):
        print("Kanal ekleniyor...")
        time.sleep(1)

        self.kanal_listesi.append(kanal)

        print("Kanal listesi: {}".format(self.kanal_listesi))

    def kanal_sec(self):

        rastgele=random.randint(0,len(self.kanal_listesi)-1)

        self.kanal=self.kanal_listesi[rastgele]
        print(self.kanal)
    
    def __len__(self):
        return len(self.kanal_listesi)

    def Connect_Youtube(self):
        while True:
            if self.youtube == "Açık":
                print("Youtube zaten açık!")
                break
            elif self.youtube == "Kapalı":
                cevap = input("Youtube'a bağlanmak istiyor musunuz? (E/H)")
                if cevap == "E" and self.ınternet == "Bağlı Değil!":
                    print("İnternet Bağlı DEĞİL!")
                    cevap2 = input("İnternete Bağlanmak İstiyor musunuz? (E/H)")
                    if cevap2 == "E":
                        print("İnternete Bağlanıyor...")
                        time.sleep(3)
                        self.ınternet = "Bağlı!"
                        print("İnternete Bağlandınız!")
                        print("Youtube'a Bağlanılıyor...")
                        time.sleep(1)
                        self.youtube = "Açık"
                        print("Youtube Açıldı!")
                        break
                    elif cevap2 == "H":
                        print("İnternete Bağlanılmadı, İnternet olmadan Youtube hizmet verememektedir! Özellik kapatılıyor...")
                        time.sleep(1)
                        break
                    else:
                        print("Lütfen E veya H harflerinden birini giriniz.")
                        continue
                elif cevap == "E" and self.ınternet == "Bağlı!":
                    print("İnternete bağlısınız, Youtube Açılıyor...")
                    self.youtube = "Açık"
                    time.sleep(1)
                    print("Youtube Açıldı!")

                elif cevap == "H":
                    print("Youtube'a bağlanmayı reddettiniz, Özellik kapatılıyor...")
                    time.sleep(1)
                    break

                else:
                    print("Lütfen E veya H harflerinden birini giriniz.")
                    continue

    def Three_dim_mode(self):
        while True:
            cevap = input("3D görüntü modunu açmak istiyor musunuz? (E/H)")
            if cevap == "E":
                if self.ucboyut_mode == "Kapalı":
                    print("3D görüntü modu açılıyor...")
                    self.ucboyut_mode = "Açık"
                    break

                else:
                    print("3D görüntü modu zaten açık.")
                    break

            elif cevap == "H":
                print("3D görüntü modunu açmayı reddettiniz, kumanda kapatılıyor.")
                break

            else:
                print("Lütfen E veya H harflerinden birini giriniz.")
                continue

    def yazılımı_guncelle(self):
        while True:
            cevap = input("Yazılım güncellemesini kontrol etmek istiyor musunuz? (E/H)")
            if cevap == "E" and  self.ınternet=="Bağlı!":
                if self.yazılım_guncelle == "Açık":
                    print("Yazılımınız güncel.")
                    break

                else:
                    print("Yazılım güncellemesi açılıyor...")
                    self.yazılım_guncelle = "Açık"
                    time.sleep(2)
                    print("Yazılım güncellendi")
                    continue
            elif cevap == "E" and self.ınternet == "Bağlı Değil!":
                print("İnternet Bağlı DEĞİL!")
                cevap2 = input("İnternete Bağlanmak İstiyor musunuz? (E/H)")
                if cevap2 == "E":
                    print("İnternete Bağlanıyor...")
                    time.sleep(3)
                    self.ınternet = "Bağlı!"
                    if self.yazılım_guncelle == "Açık":
                        print("Yazılımınız güncel.")
                        break

                    else:
                        print("Yazılım güncellemesi açılıyor...")
                        self.yazılım_guncelle = "Açık"
                        time.sleep(2)
                        print("Yazılım güncellendi")
                elif cevap2 == "H":
                    print("İnternete Bağlanılmadı, İnternet olmadan Yazılım güncellemehizmeti verilememektesir! Özellik kapatılıyor...")
                    time.sleep(1)
                    break

                else:
                    print("Lütfen E veya H harflerinden birini giriniz.")
                    continue

            elif cevap == "H":
                print("Yazılım güncellemesini açmayı reddettiniz, kumanda kapatılıyor.")
                break
            else:
                print("Lütfen E veya H harflerinden birini giriniz.")
                continue

    def __str__(self):
        return "Tv_durum: {}\ntv_ses: {}\nkanal listesi: {}\nkanal:{}\ninternet:{}\nyoutube:{}\nucboyut_mode:{}\nyazılım_guncelliği:{}".format(self.tv_durum,self.tv_ses,self.kanal_listesi,self.kanal,self.ınternet,self.youtube,self.ucboyut_mode,self.yazılım_guncelle)



print("""

        1. TV aç
        2. TV kapat
        3. Ses Ayarları
        4. Kanal Ekle
        5. Açık Kanalı Öğren
        6. Kanal Sayısı
        7. TV Bilgileri
        8.Youtube'a Bağlan
        9.3D görüntü modunu aç
        10.Yazılım Güncellemesi

Çıkmak için q'ya bas.

""")
kumanda=Kumanda()

while True:

    islem= input("İslem seçiniz")

    if islem =="q":
        print("Sonlandırılıyor")
        break 
    elif islem =="1":
        kumanda.tv_ac()
    
    elif islem=="2":
        kumanda.tv_kapat()

    elif islem=="3":
        kumanda.ses_ayari()
    
    elif islem=="4":

        kanal_isimleri=input("Kanal isimlerini lütfen , ile ayırarak giriniz.")

        x=kanal_isimleri.split(",")

        for i in x:
            kumanda.kanal_ekle(i)
    
    elif islem=="5":
        kumanda.kanal_sec()

    elif islem=="6":
        print("Kanal sayısı: ",len(kumanda))
    
    elif islem=="7":
        print(kumanda)

    elif islem=="8":
        kumanda.Connect_Youtube()

    elif islem=="9":
        kumanda.Three_dim_mode()

    elif islem == "10":
        kumanda.yazılımı_guncelle()

    else:
        print("hatalı Tuşlama")



    
