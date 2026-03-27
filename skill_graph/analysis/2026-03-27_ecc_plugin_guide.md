# Everything Claude Code (ECC) 플러그인 스킬 가이드

> 개인 참조용. ECC 플러그인에서 제공하는 스킬/명령어 정리.
> 호출: `/everything-claude-code:<skill-name>` 또는 단축 `/<skill-name>`

---

## 1. 세션 & 컨텍스트 관리

| 스킬 | 용도 |
|------|------|
| `context-budget` | 컨텍스트 윈도우 사용량 분석, 에이전트/스킬/MCP별 토큰 오버헤드 최적화 |
| `save-session` | 현재 세션 상태를 `~/.claude/session-data/`에 저장 (다음 세션에서 재개) |
| `resume-session` | 가장 최근 세션 파일 로드, 전체 컨텍스트 복원 |
| `sessions` | 세션 히스토리, 별칭, 메타데이터 관리 |
| `claw` | NanoClaw v2 REPL — 모델 라우팅, 스킬 핫로드, 브랜칭, 컴팩션, 메트릭 |
| `strategic-compact` | 논리적 전환점에서 수동 `/compact` 판단 가이드 |
| `aside` | 현재 작업 중단 없이 사이드 질문 답변 후 자동 복귀 |

## 2. 빌드 에러 해결

자동 진단 + 최소 diff로 수정. 언어별 전문 에이전트.

| 스킬 | 대상 |
|------|------|
| `build-error-resolver` | 범용 (빌드/TypeScript 에러) |
| `gradle-build` | Android, KMP Gradle 에러 |
| `java-build-resolver` | Java 컴파일, Maven/Gradle 의존성 |
| `kotlin-build` | Kotlin 컴파일, Gradle 경고 |
| `go-build` | go vet, 린터, 빌드 실패 |
| `rust-build` | cargo, borrow checker, Cargo.toml |
| `cpp-build` | CMake, 링커, 템플릿 에러 |
| `pytorch-build-resolver` | CUDA, 텐서 형태, 그래디언트, DataLoader, mixed precision |

## 3. 코드 리뷰 & 품질

### 언어별 전문 리뷰

| 스킬 | 포커스 |
|------|--------|
| `python-review` | PEP 8, 타입 힌트, 보안, 성능, Pythonic idiom |
| `go-review` | Go idiom, 동시성 패턴, 에러 처리 |
| `rust-review` | 소유권, 라이프타임, unsafe, 에러 처리 |
| `cpp-review` | 메모리 안전성, 현대 C++, 동시성 |
| `java-reviewer` | 레이어드 아키텍처, JPA, 보안, 동시성 |
| `kotlin-review` | null safety, 코루틴, Compose, 클린 아키텍처 |
| `typescript-reviewer` | 타입 안전성, 비동기 정확성, Node/웹 보안 |
| `flutter-reviewer` | 위젯, 상태 관리, Dart idiom, 접근성 |
| `database-reviewer` | PostgreSQL 쿼리 최적화, 스키마, 보안 (Supabase) |

### 범용

| 스킬 | 용도 |
|------|------|
| `code-reviewer` | 모든 코드 변경 후 품질/보안/유지보수성 검사 |
| `security-reviewer` | 인증, API, 민감 데이터, OWASP Top 10 |
| `refactor-cleaner` | 죽은 코드/중복 제거 (knip, depcheck, ts-prune) |

## 4. TDD & 테스트

테스트 먼저 작성 → 구현 → 80%+ 커버리지 검증.

| 스킬 | 프레임워크 |
|------|-----------|
| `tdd` | 범용 (모든 언어) |
| `go-test` | table-driven, subtests, benchmarks, fuzzing |
| `rust-test` | unit/integration, async, property-based, cargo-llvm-cov |
| `kotlin-test` | Kotest, MockK, 코루틴, Kover |
| `cpp-test` | GoogleTest, CTest, gcov/lcov |
| `e2e` | Playwright E2E 생성/실행, artifact 업로드 |

## 5. 멀티에이전트 오케스트레이션

| 스킬 | 용도 |
|------|------|
| `plan` | 요구사항 분석 → 단계별 구현 계획. 사용자 확인 후 코드 작성 |
| `blueprint` | 복잡 멀티세션 프로젝트 → 실행 가능한 step 계획 (dependency graph, 병렬 감지) |
| `devfleet` | Claude DevFleet로 병렬 에이전트 dispatch, worktree 격리, 진행 모니터링 |
| `orchestrate` | tmux/worktree 오케스트레이션 가이드 |
| `dmux-workflows` | dmux pane 관리자로 다중 harness 병렬 실행 |
| `team-builder` | 에이전트 팀 조합 선택 UI |
| `loop-operator` | 자율 에이전트 루프 모니터링, 안전한 개입 |
| `santa-method` | 2개 독립 리뷰 에이전트 수렴 루프 (양쪽 통과 시 출시) |

## 6. 학습 & 지식 관리

### 패턴 학습 (Instinct 시스템)

| 스킬 | 용도 |
|------|------|
| `continuous-learning-v2` | 세션에서 패턴 자동 추출, 신뢰도 점수, 프로젝트별 격리 |
| `learn-eval` | 패턴 추출 + 자가 품질 평가 + 저장 위치 판정 (Global vs Project) |
| `instinct-status` | 학습된 instinct 목록 및 신뢰도 표시 |
| `instinct-import` / `instinct-export` | instinct 가져오기/내보내기 |
| `promote` | project scope → global scope 승격 |
| `prune` | 30일+ 미사용 instinct 삭제 |
| `evolve` | instinct 분석 및 고도화 제안 |
| `projects` | 프로젝트별 instinct 통계 |

### 스킬 관리

| 스킬 | 용도 |
|------|------|
| `skill-create` | git 히스토리 분석 → SKILL.md 자동 생성 |
| `skill-health` | 스킬 포트폴리오 건강도 대시보드 |
| `skill-stocktake` | 스킬 감사 (Quick Scan / Full Stocktake) |
| `skill-comply` | 스킬이 실제 지켜지는지 행동 검증 (3가지 엄격도) |
| `rules-distill` | 스킬에서 범용 원칙 추출 → rules 파일 생성 |

## 7. 문서화 & 조사

| 스킬 | 용도 |
|------|------|
| `docs` | Context7 MCP로 최신 라이브러리 문서 조회 |
| `documentation-lookup` | 프레임워크 setup, API 레퍼런스, 코드 예제 |
| `deep-research` | firecrawl + exa 멀티소스 웹 조사 (출처 명시) |
| `exa-search` | Exa neural search — 웹, 코드, 회사, 논문 |
| `market-research` | 시장 규모, 경쟁 분석, 투자자 실사 |
| `update-docs` | README, 가이드 자동 갱신 |
| `update-codemaps` | 코드맵 자동 생성/갱신 |
| `codebase-onboarding` | 새 codebase → 구조화된 온보딩 가이드 + CLAUDE.md |

## 8. 도메인별 아키텍처 패턴

### 백엔드 프레임워크

| 스택 | 스킬들 |
|------|--------|
| Spring Boot | `springboot-patterns`, `springboot-security`, `springboot-tdd`, `springboot-verification`, `jpa-patterns` |
| Django | `django-patterns`, `django-security`, `django-tdd`, `django-verification` |
| Laravel | `laravel-patterns`, `laravel-security`, `laravel-tdd`, `laravel-verification` |
| Node/Express | `backend-patterns`, `api-design` |
| Ktor | `kotlin-ktor-patterns` |

### 프론트엔드

| 스택 | 스킬 |
|------|------|
| React/Next.js | `frontend-patterns` |
| SwiftUI | `swiftui-patterns`, `liquid-glass-design` |
| Flutter | `flutter-dart-code-review`, `compose-multiplatform-patterns` |
| Nuxt | `nuxt4-patterns` |

### 언어별 패턴

`golang-patterns`, `rust-patterns`, `kotlin-patterns`, `python-patterns`, `cpp-coding-standards`, `java-coding-standards`, `perl-patterns`, `coding-standards`(TS/JS)

### 인프라/DB

| 스킬 | 용도 |
|------|------|
| `postgres-patterns` | PostgreSQL 최적화, 스키마, Supabase |
| `clickhouse-io` | 분석형 DB 패턴 |
| `database-migrations` | 스키마 변경, 롤백, zero-downtime |
| `docker-patterns` | 로컬 개발, 보안, 네트워킹 |
| `deployment-patterns` | CI/CD, 헬스체크, 롤백 |
| `bun-runtime` | Bun 런타임 패턴 |

## 9. 평가 & 검증

| 스킬 | 용도 |
|------|------|
| `eval-harness` | 평가 기반 개발 (EDD) 프레임워크 |
| `eval` | 범용 평가 명령 |
| `agent-eval` | 코딩 에이전트 head-to-head 비교 |
| `verify` | 빌드, 타입, 린트, 테스트 종합 검증 |
| `quality-gate` | 품질 게이트 메트릭 |
| `harness-audit` | 에이전트 harness 설정 감사 |
| `harness-optimizer` | harness 신뢰도/비용/처리량 최적화 |
| `architecture-decision-records` | ADR 자동 캡처 (의사결정 이력 추적) |

## 10. 프롬프트 최적화

| 스킬 | 용도 |
|------|------|
| `prompt-optimize` / `prompt-optimizer` | 프롬프트 분석 + ECC 컴포넌트 매칭 + 최적화 버전 출력 (실행 안 함, 자문만) |

## 11. 콘텐츠 & 미디어

| 스킬 | 용도 |
|------|------|
| `content-engine` | X, LinkedIn, TikTok, YouTube, 뉴스레터 플랫폼별 콘텐츠 |
| `crosspost` | 멀티플랫폼 동시 배포 (플랫폼별 맞춤) |
| `article-writing` | 블로그, 튜토리얼, 뉴스레터 장문 콘텐츠 |
| `investor-outreach` | 투자자 콜드이메일, 팔로업 |
| `investor-materials` | 피치덱, 원페이저, 재무 모델 |
| `fal-ai-media` | fal.ai로 이미지/비디오/오디오 생성 |
| `videodb` | 비디오 수집, 프레임 추출, 시맨틱 검색, 편집 |
| `data-scraper-agent` | 공개 데이터 자동 수집 에이전트 (GitHub Actions) |
| `nutrient-document-processing` | PDF/DOCX OCR, 추출, 서명, 양식 채우기 |

## 12. 설정 & 보안

| 스킬 | 용도 |
|------|------|
| `configure-ecc` | ECC 대화형 설치 (스킬/규칙 선택, 경로 검증) |
| `security-scan` | `.claude/` 디렉토리 보안 감사 (AgentShield) |
| `security-review` | 코드 보안 체크 (인증, API, 시크릿) |

## 13. 전문 도메인 (산업별)

| 스킬 | 분야 |
|------|------|
| `logistics-exception-management` | 물류 — 배송 지연, 손상, 캐리어 분쟁 |
| `returns-reverse-logistics` | 반품 — 승인, 검사, 환불, 사기 감지 |
| `carrier-relationship-management` | 운송 — 캐리어 관리, 요금 협상, 성과 평가 |
| `inventory-demand-planning` | 재고 — 수요 예측, 안전 재고, 프로모션 lift |
| `production-scheduling` | 제조 — 스케줄링, 병목, 라인 밸런싱 |
| `quality-nonconformance` | 품질 — NCR, CAPA, SPC, 공급자 품질 |
| `energy-procurement` | 에너지 — 전기/가스 조달, 요금 최적화, PPA |
| `customs-trade-compliance` | 무역 — HS 분류, 관세, FTA, 수출입 |

---

## 현재 연구 프로젝트(FMAD)에서 유용한 스킬

```
📋 계획/설계
  /plan            — 실험 계획
  /blueprint       — 복잡 작업 분해
  /architect       — 아키텍처 결정

🔬 조사
  /deep-research   — 논문/방법론 웹 조사
  /docs            — PyTorch, sklearn 등 API 조회
  /exa-search      — 논문, 코드 검색

⚡ 병렬 실행
  /devfleet        — 병렬 에이전트 (worktree)
  /orchestrate     — 멀티에이전트 가이드

📝 코드 품질
  /python-review   — Python 코드 리뷰
  /code-reviewer   — 범용 리뷰
  /security-reviewer — 보안 검사

📊 평가
  /eval-harness    — 실험 평가 프레임워크
  /verify          — 종합 검증

🧠 학습
  /continuous-learning-v2 — 패턴 자동 추출
  /learn-eval      — 패턴 품질 평가

💾 세션
  /save-session    — 세션 저장
  /resume-session  — 세션 복원
  /context-budget  — 토큰 사용량 확인
```
