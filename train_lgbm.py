    # A) 설명용 딕셔너리
    feature_desc = {
        'GENHLTH':   "전반적인 건강 상태에 대한 자가 평가 (예: 1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', -1='Unknown')",
        'PHYSHLTH':  "지난 30일간 신체 건강이 좋지 않았던 일수 (예: 1~30 사이 정수, 0=Zero, -1=Unknown)",
        'MENTHLTH':  "지난 30일간 정신 건강(스트레스, 우울, 감정 문제 포함s던 활동을 하지 못한 일수 (예: 1~30 사이 정수, 0=Zero, -1=Unknown)",
        'POORHLTH': "지난 30일 동안 신체적 또는 정신적 건강 문제로 평소 하던 활동 (자가관리, 업무, 여가 등)을 하지 못한 일수 (예: 1~30=일수, 0=Zero, -1 = Unknown)",
        'HLTHPLN1':  "건강 보험, HMO(선불 건강관리), Medicare, 인디언 보건 서비스 등 어떤 형태의 건강 보장 제도를 갖고 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'PERSDOC2':  "개인 주치의 또는 건강관리 제공자가 있다고 생각하는지 여부 (예: 1=Yesone, 2=Morethan, 3=No, -1=Unknown)",
        'MEDCOST':   "지난 12개월 동안 비용 문제로 의사의 진료를 받지 못한 적이 있는지 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHECKUP1':  "마지막으로 정기 건강검진을 받은 시점은 언제인가? (예: 1=Within1Y, 2=Within2Y, 3=Within5Y, 4=Over5Y, 8=Never, -1=Unknown; 빈칸=null)",
        'BPMEDS':    "현재 고혈압 약을 복용 중인지 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'BLOODCHO':  "혈중 콜레스테롤 수치를 측정한 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHOLCHK':   "마지막으로 혈중 콜레스테롤 수치를 측정한 시점은 언제인가? (예: 1=Within1Y, 2=Within2Y, 3=Within5Y, 4=Over5Y, 8=Never, -1=Unknown; 빈칸=null)",
        'TOLDHI2':   "보건 전문가로부터 혈중 콜레스테롤 수치가 높다고 들은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CVDINFR4':  "보건 전문가로부터 심근경색(심장마비)을 진단받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'ASTHMA3':   "보건 전문가로부터 천식 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'ASTHNOW':   "현재 천식이 있는지 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHCSCNCR':  "보건 전문가로부터 피부암(스킨 캔서) 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHCOCNCR':  "보건 전문가로부터 피부암을 제외한 다른 암을 진단받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHCCOPD1':  "보건 전문가로부터 COPD, 폐기종, 기관지염 등 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'HAVARTH3':  "보건 전문가로부터 관절염 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'ADDEPEV2':  "보건 전문가로부터 우울 장애 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'MARITAL':   "혼인 상태 (예: 1=Mar, 2=Div, 3=Wid, 4=Sep, 5=Nev, 6=UnP, 9=Unknown)",
        'EDUCA':     "최고로 수료한 학년 또는 학력 수준 (예: 1=None, 2=Elem, 3=MidHS, 4=HS, 5=SomeCol, 6=ColGrad, 9=Unknown)",
        'RENTHOM1':  "주택 소유 형태 (예: 1=Own, 2=Rent, 3=Other; -1=Unknown)",
        'VETERAN3':  "군 복무 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'EMPLOY1':   "현재 고용 상태 (예: 1=Employed, 2=SelfEmp, 3=Unemp1Y+, 4=Unemp<1Y, 5=Homemaker, 6=Student, 7=Retired, 8=Unable, 9=Unknown)",
        'INCOME2':   "가구 연간 소득 (예: 1=<10K, 2=10-15K, 3=15-20K, 4=20-25K, 5=25-35K, 6=35-50K, 7=50-75K, 8=75K+, -1=Unknown, BLANK=null)",
        'INTERNET':  "지난 30일 이내 인터넷 사용 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'QLACTLM2':  "신체적/정신적/감정적 문제로 활동 제한 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'USEEQUIP':  "지팡이/휠체어/특수장비 사용 필요 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DECIDE':    "집중/기억/의사결정 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DIFFWALK':  "걷기/계단 오르기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DIFFDRES':  "옷 입기/목욕하기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DIFFALON':  "병원 방문/쇼핑 등 혼자 볼일 보기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'SMOKE100':  "생애 동안 담배 100개비 이상 흡연 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'STOPSMK2':  "지난 12개월 금연 시도 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'LASTSMK2':  "마지막 흡연 시기 (예: 1=Within1M, 2=Within3M, 3=Within6M, 4=Within1Y, 5=Within10Y, 7=Y10plus, 8=Never, 9/99=Unknown, 빈칸=null)",
        'USENOW3':   "현재 씹는 담배/스너프/스누스 사용 여부 (예: 1=Daily, 2=Some, 3=Never, -1=Unknown, BLANK=null)",
        'LMTJOIN3':  "현재 관절염/관절 증상으로 활동 제한 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'ARTHDIS2':  "관절염/관절 증상이 취업/업무에 미치는 영향 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'ARTHSOCL':  "지난 30일 관절염/관절 증상으로 쇼핑/영화/종교/사교 활동에 방해 정도 (예: 1=High, 2=Moderate, 3=No, -1=Unknown, 빈칸=null)",
        'JOINPAIN':  "지난 30일 관절 통증 정도 (예: 0~10 숫자, -1=Unknown, 빈칸=null)",
        'FLUSHOT6':  "지난 12개월 독감 예방접종 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'PNEUVAC3':  "폐렴 백신 접종 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'HIVTST6':   "HIV 검사 받은 적 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'HIVTSTD3':  "마지막 HIV 검사 시기 (월/년, 예: 2015년 5월=052015, -1=Unknown, BLANK=null)",
        'WHRTST10':  "마지막 HIV 검사 장소 (예: 1=Private, 2=Center, 3=Inpatient, 4=Clinic, 5=Prison, 6=DrugTx, 7=Home, 8=Other, 9=ER, -1=Unknown)",
        'PDIABTST':  "지난 3년 내 고혈당/당뇨병 검사 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'PREDIAB1':  "당뇨 전단계 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'INSULIN':   "현재 인슐린 사용 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'BLDSUGAR':  "혈당 확인 빈도 (예: 월간횟수, -1=Unknown, 0=Zero, 빈칸=null)",
        'FEETCHK2':  "발 상태 확인 빈도 (예: 월간횟수, -1=Unknown, 0=Zero, 빈칸=null)",
        'DOCTDIAB':  "지난 12개월 당뇨병 진료 횟수 (예: 1~76 직접 입력, 0=Zero, -1=Unknown, 빈칸=null)",
        'CHKHEMO3':  "수최근 12개월당화혈색소(A1C) 검사 횟수 (예: 1~76 직접 입력, 0=한 번도 없음, 7-18/99=모름, 빈칸=null)",
        'FEETCHK':   "지난 12개월 발 검사 횟수 (예: 1~76 직접 입력, 0=Zero, -1=Unknown, 빈칸=null)",
        'DIABEYE':   "당뇨 합병증(망막병증) 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CIMEMLOS':  "지난 12개월 기억력 저하/혼란 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'SXORIENT':  "성적 지향 (예: 1=이성애자, 2=레즈비언/게이, 3=양성애자, 4=기타, -1=Unknown)",
        'TRNSGNDR':  "트랜스젠더 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'MSCODE':    "거주 지역 대도시권 상태 (예: 1=MSA 중심도시, 2=MSA 주변, 3=MSA 교외, 5=비MSA, BLANK=null)",
        '_RFHLTH':   "전반적 건강 상태 (예: 1=좋음/매우좋음/보통, 2=나쁨/매우나쁨, -1=Unknown)",
        '_HCVU651':  "18~64세 응답자 건강보험 여부 (예: 1=있음, 2=없음, -1=Unknown)",
        '_CHOLCHK':  "최근 5년 내 콜레스테롤 검사 여부 (예: 1=예, 2=아니요, -1=Unknown)",
        '_RFCHOL':   "콜레스테롤 수치 높음 진단 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_LTASTH1':  "성인 후 천식 진단 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_CASTHM1':  "현재 천식 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_ASTHMS1':  "현재/과거 천식 여부 (예: 1=현재, 2=과거, 3=없음, 9=Unknown)",
        '_DRDXAR1':  "관절염 진단 여부 (예: 1=진단받음, 2=진단받지않음, -1=Unknown)",
        '_MRACE1':   "다인종 인종 분류 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=Pacific, 6=Other, 7=Multi, -1=Unknown)",
        '_HISPANC':  "히스패닉/라티노 여부 (예: 1=예, 2=아니요, -1=Unknown)",
        '_RACE':     "인종/민족 범주 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=PacificI, 6=Other, 7=Multi, 8=Hispanic, 9=Unknown)",
        '_RACEG21':  "비히스패닉 백인 여부 (예: 1=Yes, 2=No, -1=Unknown, 빈칸=null)",
        '_INCOMG':   "연간 가구 소득(재코딩) (예: 1=<15K, 2=15–25K, 3=25–35K, 4=35–50K, 5=50K+, 9=Unknown)",
        'FC60_':     "기능적 수행 능력 추정치 (예: 0–8590, 99900=Unknown)",
        '_PASTAE1':  "운동 권장 기준 충족 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_FLSHOT6':  "65세 이상 독감 예방접종 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_PNEUMO2':  "65세 이상 폐렴 백신 접종 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_AIDTST3':  "HIV 검사 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'EXRACT11':  "첫 번째 주 활동 유형 코드 (예: 1=액티브 게이밍, 2=에어로빅, 3=배낭여행, 4=배드민턴, 5=농구, 6=실내 자전거, 7=자전거, 8=카누/카약, 9=볼링, 10=복싱, 11=맨몸 체조, 12=카누 경기, 13=목공, 14=춤, 15=일립티컬/EFX, 16=낚시, 17=프리스비, 18=원예, 19=골프(전동), 20=골프(도보), 21=핸드볼, 22=하이킹, 23=하키, 24=승마, 25=대형 사냥, 26=소형 사냥, 27=인라인 스케이트, 28=조깅, 29=라크로스, 30=등산, 31=잔디 깎기, 32=패들볼, 33=집 페인트칠, 34=필라테스, 35=라켓볼, 36=낙엽 긁기, 37=달리기, 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(기계), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(장화), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=무술/가라데, 75=상체 자전거(에르고), 76=마당일, 98=기타, 77=모름, 99=응답거부, BLANK=null)",
        'EXEROFT1':  "지난 한 달 동안 이 활동(걷기, 달리기, 조깅, 수영)을 주당 또는 월당 몇 회 했습니까? (예: 월간횟수, -1=Unknown, BLANK=null)",
        'EXERHMM1':  "EXRACT11에서 선택한 운동 한 번 할 때 지속 시간 (예: 분 단위, 777=모름, 999=응답거부, BLANK=null)",
        'EXRACT21':  "지난 한 달 동안 두 번째로 많이 한 신체 활동 유형 코드 (예: 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(제설기), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(물속), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=가라데/무술, 75=상체 자전거(에르고), 76=마당일, 77=모름, 0=다른 활동 없음, 98=기타, 99=응답거부, BLANK=null)",
        'EXEROFT2':  "지난 한 달 동안 두 번째 활동을 주당/월당 몇 회 했는가? (예: 월간횟수, -1=Unknown, BLANK=null)",
        'EXERHMM2':  "두 번째 활동 한 번 할 때 지속 시간 (예: 분 단위, 777=모름, 999=응답거부, BLANK=null)",
        'DRNK3GE5':  "지난 30일 폭음 횟수 (남성: 한 번에 5잔 이상, 여성: 4잔 이상) (예: 1–76=횟수, 0=없음, -1=Unknown, BLANK=null)",
        '_RFBING5':  "폭음 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_DRNKWEK':  "주당 음주 횟수 (예: 0=음주하지 않음, 1–98999=횟수, 99900=Unknown)",
        '_RFDRHV5':  "고위험 음주 여부 (예: 1=Yes, 2=No, 9=Unknown)",
        'FTJUDA1_':  "과일 주스 1일 섭취 횟수 (예: 125=1.25회, BLANK=null)",
        'FRUTDA1_':  "과일 섭취 횟수(계산) (예: 250=2.50회, BLANK=null)",
        'BEANDAY_':  "콩류 섭취 횟수(계산) (예: 175=1.75회, BLANK=null)",
        'GRENDAY_':  "진한 녹색 채소 섭취 횟수(계산) (예: 300=3.00회, BLANK=null)",
        'ORNGDAY_':  "주황색 채소 섭취 횟수(계산) (예: 250=2.50회, BLANK=null)",
        'VEGEDA1_':  "기타 채소 섭취 횟수(계산) (예: 325=3.25회, BLANK=null)",
        '_FRUTSUM':  "전체 과일 섭취 횟수(계산) (예: 450=4.50회, BLANK=null)",
        '_VEGESUM':  "전체 채소 섭취 횟수(계산) (예: 1025=10.25회, BLANK=null)",
        '_FRTLT1':   "하루 과일 1회 이상 섭취 여부 (예: 1=Yes, 2=No, -1=Unknown, BLANK=null)",
        '_VEGLT1':   "하루 채소 1회 이상 섭취 여부 (예: 1=Yes, 2=No, -1=Unknown, BLANK=null)",
        'ACTIN11_':  "첫 번째 활동 운동 강도 (예: 0=Low, 1=Moderate, 2=Vigorous, BLANK=null)",
        'ACTIN21_':  "두 번째 활동 운동 강도 (예: 0=Low, 1=Moderate, 2=Vigorous, BLANK=null)",
        'PADUR1_':   "첫 번째 활동 주당 총 지속 시간(분) (예: 0=없음, 1–599=분, BLANK=null)",
        'PADUR2_':   "두 번째 활동 주당 총 지속 시간(분) (예: 0=없음, 1–599=분, BLANK=null)",
        'PAFREQ1_':  "첫 번째 활동 주당 빈도(계산) (예: 12000=12.000회, 99000=Unknown, BLANK=null)",
        'PAFREQ2_':  "두 번째 활동 주당 빈도(계산) (예: 12000=12.000회, 99000=Unknown, BLANK=null)",
        '_MINAC11':  "첫 번째 활동 주당 총 활동 시간(분) (예: 0=없음, 1–99999=분, BLANK=null)",
        '_MINAC21':  "두 번째 활동 주당 총 활동 시간(분) (예: 0=없음, 1–99999=분, BLANK=null)",
        'STRFREQ_':  "근력운동 주당 횟수(유산소 제외) (예: 01–99=월당 1–99회, -1=Unknown, 0=Zero, BLANK=null)",
        'PAMIN11_':  "첫 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999, BLANK=null)",
        'PAMIN21_':  "두 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999, BLANK=null)",
        'PAVIG11_':  "첫 번째 활동 중 격렬 운동 시간(분) (예: 0–99999, BLANK=null)",
        'PAVIG21_':  "두 번째 활동 중 격렬 운동 시간(분) (예: 0–99999, BLANK=null)",
        '_PAINDX1':  "운동 권장 기준 충족 지표 (예: 1=운동충족, 2=미충족, -1=Unknown)",
        '_PA300R2':  "주당 총 300분 이상 활동 여부 (예: 1=300분 이상, 2=1–299분, 3=0분, 9=Unknown)",
        '_PASTRNG':  "근력운동 권장 기준 충족 여부 (예: 1=충족, 2=미충족, -1=Unknown)",
        '_PAREC1':   "유산소+근력 운동 권장 준수 여부 (예: 1=Both, 2=AerobicOnly, 3=StrengthOnly, 4=Neither, 9=Unknown)",
        '_BMI5CAT':  "BMI 체중 상태 (예: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese, BLANK=null)",
        '_PA150R2':  "주당 150분 이상 중강도 운동 여부 (예: 1=150PLUS, 2=1to149, 3=None, 9=Unknown)",
        'BPHIGH4':   "고혈압 진단 여부 (예: 빈칸=null, 1=Yes, 3=No, -1=Unknown)",
        'CVDCRHD4':  "관상동맥 심장병 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CVDSTRK3':  "뇌졸중 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'CHCKIDNY':  "신장 질환 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DIABETE3':  "당뇨병 진단 여부 (예: 빈칸=null, 1=Yes, 3=No, -1=Unknown)",
        'SEX':       "성별 (예: 1=Male, 2=Female)",
        'PREGNANT':  "현재 임신 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'SMOKDAY2':  "흡연 여부 (예: 1='Daily', 2='Some', 3='Never', 7='Unknown', 9='Unknown', BLANK=null)",
        'ALCDAY5':   "지난 30일 음주 일 수 (예: 월간횟수, -1=Unknown, 0=Zero, BLANK=null)",
        'AVEDRNK2':  "지난 30일 평균 일 음주량 (예: 1–7=잔 수, -1=Unknown, BLANK=null)",
        'MAXDRNKS':  "지난 30일 최대 음주량 (예: 1–76=잔 수, -1=Unknown, BLANK=null)",
        'FRUITJU1':  "지난 한 달 과일 주스 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'FRUIT1':    "지난 한 달 과일 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'FVBEANS':   "지난 한 달 콩류 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'FVGREEN':   "지난 한 달 녹색 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'FVORANG':   "지난 한 달 주황색 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'VEGETAB1':  "지난 한 달 기타 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, -1=Unknown, BLANK=null)",
        'EXERANY2':  "지난 30일 운동 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'STRENGTH':  "근력운동 횟수 (예: 월간횟수, -1=Unknown, 0=Zero, BLANK=null)",
        '_RFHYPE5':  "고혈압 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_MICHD':    "관상동맥 심장병(CHD) 또는 심근경색(MI) 여부 (예: 빈칸=null, 1=Yes, 2=No)",
        'HTM4':      "키(미터) (예: 91–244, 빈칸=null)",
        'WTKG3':     "몸무게(kg) (예: 30–200, 99999=Unknown)",
        '_BMI5':     "체질량지수(BMI) (예: 18.5–40)",
        '_SMOKER3':  "현재 흡연 여부 (예: 1='Daily', 2='Some', 3='Former', 4='Never', 9='Unknown')",
        'DRNKANY5':  "최근 30일간 음주 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        'DROCDY3_':  "최근 일주일간 평균 음주량 (예: 정수(3,7,10,71)=하루 평균 잔 수, 900=Unknown, 빈칸=null)",
        '_FRUITEX':  "과일 섭취 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_VEGETEX':  "채소 섭취 여부 (예: 빈칸=null, 1=Yes, 2=No, -1=Unknown)",
        '_PACAT1':   "신체 활동 수준 범주 (예: 1=VeryActive, 2=Active, 3=Insufficient, 4=Inactive, 9=Unknown)",
        '_AGEG5YR':  "나이 5세 단위 그룹 (예: 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+)"
    }
    
    # 입력 루프
    # ────────────────────────────────────────────────────────────────────
    # 9) 사용자 입력 루프: “모든 컬럼”과 “설명”을 출력한 뒤, 값을 차례대로 입력받음
    # ────────────────────────────────────────────────────────────────────
    new_data = {}

    for col in feature_cols:
        desc = feature_desc.get(col, "설명 없음")
        prompt = f"- [{col}] {desc}\n  → "
        val = input(prompt).strip()

    # ⚠️ "Unknown" 문자열 입력 → NaN 처리
        if val.lower() in [''] or val in ['NaN']:
            num = pd.NA
        else:
            try:
                num = int(val)
            except:
                num = pd.NA
                
            new_data[col] = num