from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from model import AnswerPredictor

MODEL_PATH = "/content/drive/MyDrive/project/"
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

context= 'سطح سر و صدای سطح پایین وینیل نسبت به شلاک فراموش نشد و همچنین دوام آن فراموش نشد. در اواخر دهه 1930 ، تبلیغات رادیویی و برنامه های رادیویی از قبل ضبط شده برای سوار بر دیسک ها شروع به مهر زدن در وینیل می کردند ، بنابراین از طریق پست نمی شکنند. در اواسط دهه 1940 ، به همین دلیل ، نسخه های ویژه دی وی دی ضبط شده از وینیل نیز شروع شد. اینها همه دور در دقیقه 78 بودند. در طول جنگ جهانی دوم و بعد از آن ، هنگامی که منابع شلاک بسیار محدود بود ، برخی از رکوردهای 78 دور در دقیقه به جای شلاک ، در وینیل تحت فشار قرار گرفتند ، به ویژه رکوردهای شش دقیقه ای 12 اینچ (30 سانتی متر) 78 دور در دقیقه که توسط V-Disc تولید شده و برای توزیع به یونایتد تولید شده است. سربازان ایالات متحده در جنگ جهانی دوم. در دهه 1940 ، رونویسی رادیو که معمولاً در رکوردهای 16 اینچی بود ، اما گاهی 12 اینچ بود ، همیشه از وینیل ساخته می شد ، اما در 33 ⁄ 1 دور در دقیقه بریده می شد. رونویسی های کوتاهتر اغلب در 78 دور در دقیقه بریده می شدند.'
question= 'کدام ماده با دوام تر ، شلاک یا وینیل بود؟'
predictor = AnswerPredictor(model, tokenizer, device='cpu', n_best=10, no_answer=True)
preds = predictor(question, context, batch_size=1)
pred = preds[0]['text'].strip()
print(pred)