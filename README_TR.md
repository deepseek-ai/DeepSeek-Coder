# DeepSeek-Coder — Türkçe Rehber (tugi06 çatallaması)

Bu depo, DeepSeek'in açık kaynak kod modeli **DeepSeek-Coder**'ın bir kopyasıdır (çatallaması). Orijinal İngilizce açıklama [README.md](README.md) dosyasındadır. Bu dosya, depoyu Türkçe olarak tanıtır.

## Bu depo ne işe yarar?
- **DeepSeek-Coder modellerini denemek:** Kod yazma, kod tamamlama ve kod hakkında soru-cevap için kullanılır.
- **Modeli kendi verinle eğitmek (ince ayar):** Hazır betikler `finetune/` klasöründedir.
- **Model başarısını ölçmek:** Değerlendirme betikleri `Evaluation/` klasöründedir.

> Projelerim (ör. GüneşSuTV) bu depoda **tutulmaz**. Onlar ayrı ve gizli depolardadır; DeepSeek'e oradaki dosyalar verilir.

## Klasörler

| Klasör / dosya | İçerik |
|---|---|
| `README.md` | DeepSeek'in orijinal İngilizce açıklaması (model boyutları, sonuçlar, örnekler) |
| `demo/` | Modeli tarayıcıda denemek için küçük bir sohbet uygulaması |
| `finetune/` | İnce ayar betiği (`finetune_deepseekcoder.py`) ve kendi gereksinim listesi |
| `Evaluation/` | HumanEval, MBPP, DS-1000, PAL-Math, LeetCode gibi kıyaslama testleri |
| `pictures/` | README'deki görseller |
| `requirements.txt` | Çalıştırmak için gereken Python paketleri |
| `LICENSE-CODE`, `LICENSE-MODEL` | Kod lisansı (MIT) ve model kullanım lisansı |

## Hızlı başlangıç

**1. Gerekli paketleri kur:**
```bash
pip install -r requirements.txt
```

**2. Kod tamamlama örneği:** Bu örnek Hugging Face'ten küçük bir modeli indirir. Model birkaç GB yer tutar. Bilgisayarında ekran kartı yoksa `.cuda()` ifadelerini sil; o zaman çok yavaş çalışır.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
ad = "deepseek-ai/deepseek-coder-1.3b-instruct"
tok = AutoTokenizer.from_pretrained(ad, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ad, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
mesaj = [{"role": "user", "content": "Python'da bir listeyi tersine çeviren fonksiyon yaz."}]
girdi = tok.apply_chat_template(mesaj, add_generation_prompt=True, return_tensors="pt").to(model.device)
cikti = model.generate(girdi, max_new_tokens=256, do_sample=False, eos_token_id=tok.eos_token_id)
print(tok.decode(cikti[0][len(girdi[0]):], skip_special_tokens=True))
```

**3. Daha büyük modeller:** 6.7B ve 33B modeller daha iyi sonuç verir ama güçlü bir ekran kartı ister. Model listesi için [README.md](README.md) dosyasına bak.

## DeepSeek'e iş verirken
- **Talimatı açık yaz:** Proje yapısını, kuralları ve yapılacakları tek bir metinde ver.
  - Örnek metin: `GunesSuTV-DeepSeek` deposundaki `BASLA.md` dosyası.
- **Kontrol et:**
  - Testleri çalıştırt.
  - Değişiklikleri ekran görüntüsüyle göstert.
  - "TV'de denendi" ile "bilgisayarda test edildi" arasındaki farkı sor.
- **Gizli bilgi verme:** Şifre, API anahtarı ve imza anahtarı (keystore) gibi gizli bilgileri DeepSeek'e verme.

## Lisans
- **Kod:** MIT lisansı ([LICENSE-CODE](LICENSE-CODE)).
- **Modeller:** DeepSeek model lisansı ([LICENSE-MODEL](LICENSE-MODEL)). Ticari kullanıma izin verir, ama lisanstaki kısıtlamalara uymak gerekir.
