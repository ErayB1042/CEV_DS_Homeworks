class Hayvanlar():
    def __init__(self, isim,tür, boy, uzunluk, ağırlık, habitat, beslenmeşekli):
        self.isim = isim
        self.tür=tür
        self.boy = boy
        self.uzunluk = uzunluk
        self.ağırlık = ağırlık
        self.habitat = habitat
        self.beslenmeşekli = beslenmeşekli

    def bilgileri_goster(self):
        print("Canlıların bilgileri...")
        print("İsim : {}\nTür: {}\nBoy: {}\n Uzunluk: {}\n Ağırlık: {}\n Habitat: {}\nBeslenme şekli: {}\n".format(self.isim,self.tür, self.boy, self.uzunluk, self.ağırlık, self.habitat, self.beslenmeşekli))


class Dinozor(Hayvanlar):
    def __init__(self,isim,tür,boy,uzunluk,ağırlık,habitat,beslenmeşekli,hız,cenegucu):
        super().__init__(isim,tür,boy,uzunluk,ağırlık,habitat,beslenmeşekli)
        self.hız=hız
        self.cenegucu=cenegucu

    def __str__(self):
        return "İsim : {}\nTür: {}\nBoy: {}\n Uzunluk: {}\n Ağırlık: {}\n Habitat: {}\nBeslenme şekli: {}\nHız: {}\nÇene Gücü:{}".format(self.isim,self.tür, self.boy, self.uzunluk, self.ağırlık, self.habitat, self.beslenmeşekli,self.hız,self.cenegucu)



Dino1= Dinozor("Mehmet","T-REX",2,5,100,"Savanna","Etobur",2,15)
print(Dino1)


class Kartal(Hayvanlar):
    def __init__(self,isim,tür,boy,uzunluk,ağırlık,habitat,beslenmeşekli,cinsiyet="Erkek",yuva_yapıcı="Hayır",yavru_sayısı=0):
        super().__init__(isim,tür,boy,uzunluk,ağırlık,habitat,beslenmeşekli)

        self.cinsiyet=cinsiyet
        self.yuva_yapıcı=yuva_yapıcı
        self.yavru_sayısı=yavru_sayısı

    def cinsiyet_değiştir(self):
        if self.cinsiyet=="Erkek":
            self.cinsiyet = "Dişi"
            self.yuva_yapıcı = "Evet"
        elif self.cinsiyet == "Dişi":
            self.cinsiyet = "Erkek"
            self.yuva_yapıcı = "Hayır"

    def __str__(self):
        return "İsim : {}\nTür: {}\nBoy: {}\n Uzunluk: {}\n Ağırlık: {}\n Habitat: {}\nBeslenme şekli: {}\nCinsiyet: {}\nYuva Yapıcı:{}\nYavru Sayısı: {}".format(
            self.isim, self.tür, self.boy, self.uzunluk, self.ağırlık, self.habitat, self.beslenmeşekli,self.cinsiyet,self.yuva_yapıcı,self.yavru_sayısı )

kartal1=Kartal("Bengisu","Kartal",1,1.5,15,"Dağlar","Etobur","Dişi","Evet",3)
print(kartal1)

class Balıklar(Hayvanlar):
    def __init__(self,isim,tür,boy,uzunluk,ağırlık,habitat="Deniz",beslenmeşekli="Otobur",cinsiyet="Erkek",yavru_sayısı=10,bulundugusu="Tatlı"):
        super().__init__(isim, tür, boy,uzunluk,ağırlık,habitat,beslenmeşekli)
        self.cinsiyet = cinsiyet
        self.yavru_sayısı = yavru_sayısı
        self.bulundugusu=bulundugusu

    def Su_tipi(self):
        print(self.bulundugusu)

    def __str__(self):
        return "İsim : {}\nTür: {}\nBoy: {}\nUzunluk: {}\n Ağırlık: {}\nHabitat: {}\nBulunduğu Su Tipi: {}\nCinsiyet: {}\nBeslenme Şekli:{}".format(
            self.isim, self.tür, self.boy, self.uzunluk, self.ağırlık,self.habitat, self.bulundugusu,self.cinsiyet,
            self.yavru_sayısı,self.beslenmeşekli)

Köpekbalığı1 = Balıklar("Ozan","Köpekbalığı",5,10,"Deniz","Erkek",5,"Tuzlu")
print(Köpekbalığı1)

