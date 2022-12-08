import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, math
import numpy as np
from loguru import logger
from PIL import Image, ImageFont, ImageDraw
import cv2
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE))
sys.path.append(BASE)
from text_recognition.rec_model.hangulnet.hangulnet import HangulNet
from text_recognition.rec_model.dtr.recnet import Model
from ocr.preprocess import preprocess_for_recognition
from text_recognition.utils.label_convert import CTCLabelConverter, AttnLabelConverter
import torchvision.transforms as transforms
MODEL_PATH='/home/ubuntu/user/jihye.lee/ocr_exp_v1/flask_serve/model'
class Recog_CFG:
    IMG_H=32
    IMG_W=128
    CHARACTER='0123456789가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!'
    PRETRAINED_MODEL=os.path.join(MODEL_PATH, 'best_accuracy_v2.pth')
    HIDDEN_DIM=256
    MAX_LENGTH=25
    PREDICTION='Attn'

class TextRecognizer(object):
    def __init__(self):
        super(TextRecognizer, self).__init__()
        # self.model = HangulNet()
        self.class_n = len(Recog_CFG.CHARACTER)
        self.model = Model(class_n=self.class_n).cuda()
        if Recog_CFG.PRETRAINED_MODEL != '':
            try:
                self.model.load_state_dict(torch.load(Recog_CFG.PRETRAINED_MODEL))
            except:
                model_dict = self.model.state_dict()
                pretrained = torch.load(Recog_CFG.PRETRAINED_MODEL)
                for key, value in model_dict.items():
                    model_dict[key] = pretrained[f"module.{key}"]
                self.model.load_state_dict(model_dict)
        self.converter = AttnLabelConverter(Recog_CFG.CHARACTER)

    def run(self, bbox, image):
        text_box = bbox['box']
        box_image = image.copy()
        fontpath = "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"
        answer_text = {}
        font = ImageFont.truetype(fontpath, 13)

        copy_image = Image.fromarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
       
        cv2.imwrite('/home/ubuntu/user/jihye.lee/ocr_exp_v1/flask_serve/img_test_res/img.png', image)
        draw = ImageDraw.Draw(copy_image)
        for idx, box in enumerate(text_box):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            croped_img = image[y1:y2, x1:x2]
            if idx < 10:
                text = self.predict(croped_img, save=True, idx=idx)
            else:
                text = self.predict(croped_img, save=False)
            if text is None:
                continue
            ## 각각의 bounding box의 좌표와 그에 해당하는 인식된 text정보를 json 형태로 보내줌
            answer_text[idx] = {
                'box': [x1,y1,x2,y2],
                'text': text
            }
            if text.strip(' ') != '':
                cv2.rectangle(box_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
            draw.text((x1,y1-20),str(text),(0,0,255), font=font)
        return box_image,copy_image, answer_text

    def predict(self, image, save=False, idx=0):
        if image.size == 0:
            return None
        
        h,w = image.shape
        ## (1) Recognition을 위해서 bounding box 그대로 잘라서 모델에 넣어줄때에도
        # width와 height의 원래 비율을 맞춰 주어야 한다.
        ratio = w / float(h)
        if math.ceil(Recog_CFG.IMG_H * ratio) > Recog_CFG.IMG_W:
            resized_w = Recog_CFG.IMG_W
        else:
            resized_w = math.ceil(Recog_CFG.IMG_H * ratio)
        image = cv2.resize(image, (resized_w, Recog_CFG.IMG_H))
        ## (2) 이미지의 이진화를 진행해서 보다 선명하게 글씨 부분을 인식하도록 한다.
        _, image = cv2.threshold(image, 0, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        if save:
            cv2.imwrite(os.path.join('/home/ubuntu/user/jihye.lee/ocr_exp_v1/flask_serve/img_test_res', f'img_{idx}.png'), image)

        ## (3) 학습 시켰을 때와 동일하게 image normalization을 한다.
        aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,], std = [0.5,])
        ])
        tensor_image = aug(image).unsqueeze(0).cuda()
        self.model.eval()
        length_for_pred = torch.IntTensor([Recog_CFG.MAX_LENGTH]).cuda()
        text_for_pred = torch.LongTensor(1, Recog_CFG.MAX_LENGTH + 1).fill_(0).cuda()
        
        if 'CTC' in Recog_CFG.PREDICTION:
            with torch.no_grad():
                preds = self.model(tensor_image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            with torch.no_grad():
                preds = self.model(tensor_image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            
            try:
                pred_eos = preds_str[0].find('[s]')
                preds_str = preds_str[0][:pred_eos]
            except:
                preds_str = preds_str[0]
        logger.info(preds_str)
        return preds_str


