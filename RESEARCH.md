# 조사 및 구현 근거

확인일: 2026-10-31. 원본 설치 폴더는 읽기만 하고, 작업과 결과물은 이 저장소 안에 분리했다.

## 원본 구조와 선택

- [TartarusSolver v0.7.20](https://github.com/glcoder/TartarusSolver/tree/v0.7.20): MIT 공개 소스가 원본 버전과 일치한다. .NET 9 WPF, Windows 화면 캡처, ONNX 이미지 분류와 PaddleOCR, 게임 로그, 보상 탐색 엔진을 사용한다. 소스가 있으므로 REA 설치나 실행 파일 역컴파일은 필요하지 않았다.
- [DinoRaos/vorax-solver-rs](https://github.com/DinoRaos/vorax-solver-rs): MIT 커밋 `9aabc20`을 도입했다. Rust의 화면 인식, Windows OCR, 클릭 통과 오버레이와 미래 상태 시뮬레이션을 재사용한다. 원형 TypeScript 엔진과 상태 전이, 난수 순서, 추천 결과를 비교하는 회귀 자료가 포함되어 있다.
- Vorax 엔진의 출처는 tidewarden 프로젝트의 마지막 MIT 커밋 `7b4c9ba`까지다. 이후 재배포/수정을 제한한 코드는 사용하지 않았다. 라이선스를 확인할 수 없는 실행 파일과 무거운 Electron 래퍼도 채택하지 않았다.

## 게임 자료

- [TLIDB 한국어 시즌 정보](https://tlidb.com/ko/Vorax_Season), [영어 시즌 정보](https://tlidb.com/en/Vorax_Season): 블러드 러스트, 수술 준비, 배양 탱크와 수술 도구 명칭.
- [TLIDB 수술 도구](https://tlidb.com/en/Surgical_Instrument), [한국어 용어 연결](https://tlidb.com/ko/Hyperlink), [영어 용어 연결](https://tlidb.com/en/Hyperlink): 카드 및 효과 명칭 대조.
- [VoraxRoom](https://github.com/madstone290/VoraxRoom): 한국어 게임 용어 대조용으로만 참고했다. 이 프로젝트의 코드는 복사하지 않았다.
- 2026-10-31 실제 사용자 한국어 1920x1080 수술 준비 화면을 읽어 금색 카드 제목, 새로고침 숫자 위치와 탱크 수치를 검증했다. 개인 설정과 로그는 배포물에 포함하지 않는다. 사용자가 요청한 README 예시에는 계정 정보가 없는 선정된 게임/프로그램 화면만 사용한다.

## 추천의 의미

타르타로스는 원본의 보상 가치, 목표, 남은 일수, 현재 도구와 보드를 사용한다. 블러드 러스트는 인식한 보드에서 가능한 카드/대상/새로고침별 미래를 시뮬레이션하고 840,000 활성 목표에 도달할 가능성을 비교한다. 확률과 모델에 근거한 추천이며 수학적으로 증명된 전역 최적해나 실제 보상 보장이 아니다. 미래 카드의 실제 순서나 시장 가격은 알지 못한다.

인턴 의사 세라일은 도구 새로고침 초기 횟수를 바꾼다. 미보유/1단계 0회, 2/3단계 1회, 4/5/6단계 2회를 기본값으로 사용한다. 화면에서 확실하게 읽은 실제 남은 횟수를 우선하므로 이미 소모한 횟수를 펫 단계로 복원하지 않는다. 수술 플랜 11~13라운드는 캐릭터 생존력에 따라 판단해야 하므로 자동 확정 추천에서 제외한다.

## 실행 비용과 배포

게임 로그에 따라 캡처/계산하며, 읽기에 실패하면 간격을 두고 재시도한다. 자동 모드에서 탭은 설정 화면이며 콘텐츠 전환은 게임 로그로 판단한다. 블러드 러스트 준비 화면을 잠시 닫으면 표시와 재시도를 멈추고 모델/확정 도구 이력은 유지한다. 타르타로스 진입이나 게임 종료 시 블러드 러스트 엔진을 종료한다. Vorax 계산은 기본 최대 8개 스레드와 선택지당 256/512/1,024회 탐색을 제공한다. 게임 메모리 접근이나 게임 입력 자동화는 사용하지 않는다.

원본과 설정을 분리했다. 원본 업데이트/오류 전송은 비활성화하고, 동반 앱의 업데이트 확인은 kinam-ssafy/torchlight-overlay의 공개 릴리스만 사용한다. 라이선스/종속성 고지와 독립 실행형 .NET을 동봉한다. 패키징은 개인 설정, 진단 이미지, 로그, 디버그 심볼을 제외하고 파일별 SHA-256 목록 및 ZIP 체크섬을 생성한다.

## 한국어 명칭 전수 대조 (2026-11-01)

설치된 게임의 패치 6249411에 포함된 영어/한국어 언어 자료를 읽기 전용으로 대조했다. 동일한 `map_gameplay_season_plague_card_effect|name|ID`를 기준으로 대응시켰다. 추출 자료의 압축 전 SHA-1과 해제 크기를 확인했으며, 원본 언어 자료와 추출 도구는 배포하지 않는다.

| ID | 솔버 식별자 | 한국어 원문 |
|---|---|---|
| 401 | Plague Quarter Peat Poultice | 감염 구역 토탄 드레싱 |
| 402 | Compound Revitalizing Pill | 복방 활력 알약 |
| 403 | Insect Attractant | 벌레 유인제 |
| 404 | Purest Holy Water | 순수한 성수 |
| 405 | Fast-Acting Heart Tonic | 즉효 강심제 |
| 406 | Rousing Salts | 의식 회복용 소금 |

404/405의 현재 영문 표기는 각각 The Holiest Water / Fast-Acting Cardiac Stimulant다. 효과 설명으로 기존 솔버 식별자와 대조했으며 엔진의 효과/RNG는 변경하지 않았다. 원본의 범용 Xenopanacea 식별자는 현행 두 카드의 공통 명칭인 '이종 만능약'으로 표시한다. 이것만 읽힌 경우 변종을 구분하지 못하므로 확정하지 않는 검사를 유지한다. 지원 식별자 85개 전체의 한국어 누락을 검사하고, 실제 카드 제목 84개는 경쟁 후보와의 점수 차이까지 검사한다.

파일 형식 확인에 [repak](https://github.com/trumank/repak), 압축 해제에 [ooz](https://github.com/powzix/ooz), 게임별 컨테이너 정보는 [Gildor의 Torchlight Infinite 자료](https://www.gildor.org/smf/index.php?topic=8083.0)를 참고했다. 이 도구들은 런타임 의존성이 아니다.
