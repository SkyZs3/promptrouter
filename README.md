# PromptRouter: Abonelik Tabanlı Yapay Zekâ Yönlendirici

Kullanıcının sahip olduğu yapay zekâ hesaplarına (ücretsiz/ücretli) göre, yazdığı soru için
en uygun yapay zekâyı ve limit dolarsa yedeğini öneren bir Streamlit uygulaması.
Ücretli API kullanılmaz. Sınıflandırma varsayılan olarak anahtar kelime filtresiyle yapılır (internetsiz, ücretsiz); isteğe bağlı olarak ücretsiz bir API anahtarıyla (Gemini/Groq) yapay zekâ ile de yapılabilir, anahtar yoksa ya da internet yoksa filtre devreye girer.

## Kurulum ve çalıştırma
```
pip install -r requirements.txt
streamlit run app.py
```
Tarayıcıda açılan adres (aynı ağdan "Network URL" ile de erişilebilir) uygulamayı gösterir.
`app.py` ve `benchmark.csv` aynı klasörde olmalıdır.

## Nasıl çalışır
1. Sol menüden her yapay zekâ için hesap türü seçilir: Yok / Ücretsiz / Ücretli.
2. Soru yazılıp "Öner"e basılır. Soru anahtar kelimelere göre bir kategoriye ayrılır:
   Yazılım, Matematik ve hesaplama, Bilim, Yaratıcı yazı, Genel veya karma (Bilim + hesaplama).
   Kategori yanlışsa elle değiştirilebilir.
3. `benchmark.csv` içindeki puanlar pandas ile filtrelenir; kullanıcının hesap türüne uygun
   model puanına göre sıralanır. İlk sıra öneri, ikinci sıra yedektir.
4. "Siteyi aç" düğmesi ilgili siteyi açar; soru kutudaki kopyala simgesiyle kopyalanır.
   "Limitim doldu" denirse o yapay zekâ çıkarılıp sıradaki önerilir.
5. Grafikler: seçilen kategorideki puanlar (çubuk), tüm yapay zekâ/kategori ısı haritası,
   ve oturumdaki isteklerin kategori/öneri dağılımı (Matplotlib + Seaborn).

## Yapay zekâ ile sınıflandırma (isteğe bağlı)
- Sol menüde "Soru kategorisi nasıl belirlensin?" seçeneği vardır: Otomatik (Groq, olmazsa Gemini, olmazsa kelime filtresi), Sadece Gemini, Sadece Groq ya da Sadece kelime filtresi (internetsiz).
- Yapay zekâ seçenekleri için ücretsiz bir API anahtarı gerekir: `GEMINI_API_KEY` (Google AI Studio) ve/veya `GROQ_API_KEY` (Groq). Anahtar ortam değişkeni ya da `.streamlit/secrets.toml` ile verilir; kodun içinde ve teslim dosyalarında yoktur.
- İnternet yoksa, yavaşsa (8 saniye zaman aşımı), kota dolduysa ya da cevap anlaşılmazsa uygulama sıradaki sağlayıcıya, o da olmazsa kelime filtresine döner ve nedenini ekranda yazar.
- Yapay zekâ seçeneklerinde soru metni Google/Groq sunucusuna gönderilir. Aynı soru için başarılı sonuç oturum boyunca saklanır.

## Hesaplar
- Sol menüden kullanıcı adı ve şifreyle kayıt olunup giriş yapılabilir. Giriş yapmadan da kullanılır (misafir), ama bilgi kaydedilmez.
- Girişli kullanıcının abonelik seçimleri (hangi yapay zekâda hangi hesap) ve istek geçmişi `promptrouter.db` (SQLite) dosyasında saklanır; tekrar girince otomatik yüklenir.
- Girişli kullanıcı, "Hesap kullanım geçmişi" bölümünde hesabın oluşturulma tarihini, giriş zamanlarını ve hatalı şifre denemelerini görür.
- Şifreler düz yazı saklanmaz: rastgele tuz + PBKDF2-SHA256 ile özetlenir.
- `promptrouter.db` ilk çalıştırmada otomatik oluşur; teslim dosyalarına eklenmesine gerek yoktur.

## Dosyalar
- `app.py`: uygulama kodu
- `benchmark.csv`: puan tablosu (kategori, plan, firma, model, puan, hata_payi, tarih, kaynak)
- `requirements.txt`: gerekli kütüphaneler
- `promptrouter.db`: kullanıcı ve geçmiş veritabanı (çalışırken otomatik oluşur)

## Veri kaynağı ve sınırlamalar
- Puanlar arena.ai Text Arena'dan, 23.10.2026 tarihinde elle alındı (kullanıcı oylarına dayanır).
- Her firmanın ücretli satırı o kategorideki en yüksek puanlı modelidir.
- Ücretsiz satırlar, ücretsiz katmanda kullanıldığı varsayılan modellerin puanıdır; bu eşleştirme
  tahmindir. Grok Ücretsiz için güvenilir veri bulunamadığından sıralamaya katılmaz.
- İlk iki yapay zekânın puan farkı çoğu zaman hata payı içindedir; uygulama bunu not olarak gösterir.
- Anahtar kelime filtresi yorum yapmaz; tanıyamadığı soruyu "Genel" sayar.
- Veri canlı değildir, `benchmark.csv` elle güncellenir.
