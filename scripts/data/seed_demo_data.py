#!/usr/bin/env python3
"""Seed demo conversation data into ConversationStore for testing.

Generates realistic multi-turn conversations spanning code generation,
creative writing, math help, and general Q&A, each with simulated user feedback.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainable_openclaw.logging.conversation_store import ConversationStore

# --- Realistic multi-turn conversations ---

DEMO_CONVERSATIONS = [
    {
        "user_id": "coder_zhang",
        "model": "qwen3-4b",
        "messages": [
            ("user", "用Python写一个函数，合并两个有序列表并去重"),
            ("assistant", "def merge_and_dedup(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            if not result or result[-1] != a[i]:\n                result.append(a[i])\n            i += 1\n        else:\n            if not result or result[-1] != b[j]:\n                result.append(b[j])\n            j += 1\n    while i < len(a):\n        if not result or result[-1] != a[i]:\n            result.append(a[i])\n        i += 1\n    while j < len(b):\n        if not result or result[-1] != b[j]:\n            result.append(b[j])\n        j += 1\n    return result",
             {"quality_score": 0.85, "feedback": "代码正确但变量名a,b不够表意，建议用list1,list2"}),
            ("user", "能不能加上类型注解和文档字符串？"),
            ("assistant", "def merge_and_dedup(\n    seq1: list[int], seq2: list[int]\n) -> list[int]:\n    \"\"\"Merge two sorted lists into one sorted list without duplicates.\n\n    Args:\n        seq1: first sorted list of integers\n        seq2: second sorted list of integers\n\n    Returns:\n        Merged and deduplicated sorted list\n    \"\"\"\n    result: list[int] = []\n    i = j = 0\n    while i < len(seq1) and j < len(seq2):\n        if seq1[i] < seq2[j]:\n            if not result or result[-1] != seq1[i]:\n                result.append(seq1[i])\n            i += 1\n        else:\n            if not result or result[-1] != seq2[j]:\n                result.append(seq2[j])\n            j += 1\n    while i < len(seq1):\n        if not result or result[-1] != seq1[i]:\n            result.append(seq1[i])\n        i += 1\n    while j < len(seq2):\n        if not result or result[-1] != seq2[j]:\n            result.append(seq2[j])\n        j += 1\n    return result",
             {"quality_score": 0.92, "feedback": "类型注解完整，文档清晰，变量命名改进了"}),
        ],
    },
    {
        "user_id": "coder_zhang",
        "model": "qwen3-4b",
        "messages": [
            ("user", "解释一下Python的GIL是什么"),
            ("assistant", "GIL (Global Interpreter Lock) 是CPython解释器的一个互斥锁，确保同一时刻只有一个线程在Python解释器中运行。这导致Python多线程无法充分利用多核CPU，但简化了内存管理。对于IO密集型任务可以用多线程，CPU密集型任务建议用多进程(multiprocessing)。",
             {"quality_score": 0.78, "feedback": "解释清楚但过于简短，可以补充历史原因和GIL对日常开发的实际影响"}),
            ("user", "那asyncio和GIL有什么关系？"),
            ("assistant", "asyncio是单线程的事件循环，不绕开GIL。但因为它是单线程的，所以不受GIL多线程切换的开销影响。asyncio通过协程切换实现并发，适合IO密集型任务。多线程受GIL限制但可以用C扩展释放GIL；asyncio没有这个问题但对CPU密集型任务也无帮助。",
             {"quality_score": 0.88, "feedback": "讲清楚了asyncio与GIL的关系，举例恰当"}),
        ],
    },
    {
        "user_id": "writer_li",
        "model": "qwen3-4b",
        "messages": [
            ("user", "帮我用古风文笔写一段描述秋日黄昏的散文"),
            ("assistant", "西风渐紧，雁字回时，已是暮秋时节。斜阳脉脉，将最后一抹金辉洒在远山，那山便像镀了一层薄薄的金箔，映着天际的流霞。庭前老槐，叶落殆尽，剩下虬曲的枝干在夕照中投下长长的影。偶有寒鸦数点，掠过檐角，留下几声凄清的啼鸣。我独坐石阶，看这满目萧瑟，心中却生出一种说不出的宁静。",
             {"quality_score": 0.95, "feedback": "文笔优美，意境深远，可媲美专业散文"}),
            ("user", "续写一段，加入人物对话和情感"),
            ("assistant", "正出神间，忽闻身后脚步声轻响。回首望去，却是一位白发老翁，拄杖徐行，也来此处看这落日余晖。\n\n"老丈，您也常来此处？"我起身相询。\n\n老翁微微一笑，目光投向远方："五十年了。这山、这树、这夕阳，俱是旧时模样。只是看山的人，一日老似一日。"\n\n我默然。他又道："少年人，你可知道，人生最怕什么？不是岁月催人老，而是人在时光里，忘了自己为何而来。"\n\n说罢，他拄杖缓步而去，身影渐行渐远，终没入苍茫暮色之中。我望着他离去的方向，只觉心中某个角落，被轻轻触碰了。",
             {"quality_score": 0.91, "feedback": "加入对话后情感层次丰富了，老翁的台词很有哲理"}),
        ],
    },
    {
        "user_id": "math_student",
        "model": "qwen3-4b",
        "messages": [
            ("user", "求函数 f(x) = x^3 - 3x^2 + 2x 在区间 [0, 3] 上的极值"),
            ("assistant", "逐步求解：\n1. 求导: f'(x) = 3x^2 - 6x + 2\n2. 令 f'(x) = 0: 3x^2 - 6x + 2 = 0 → x = 1 ± 1/√3 ≈ 1.577 或 0.423\n3. 比较: f(0)=0, f(0.423)≈0.385, f(1.577)≈-0.385, f(3)=6\n4. 在[0,3]上: 最大值=f(3)=6, 最小值=f(1.577)≈-0.385",
             {"quality_score": 0.72, "feedback": "计算正确但第三步比较时数值精度不够，应该保留根号形式"}),
            ("user", "能精确表达吗？不要近似值"),
            ("assistant", "用精确形式：\n1. f'(x) = 3x^2 - 6x + 2 = 0\n2. 解得: x = 1 ± √3/3\n3. f(0) = 0\n   f(1-√3/3) = (1-√3/3)^3 - 3(1-√3/3)^2 + 2(1-√3/3) = 2√3/9\n   f(1+√3/3) = (1+√3/3)^3 - 3(1+√3/3)^2 + 2(1+√3/3) = -2√3/9\n   f(3) = 6\n4. 最大值 = f(3) = 6; 最小值 = f(1+√3/3) = -2√3/9",
             {"quality_score": 0.94, "feedback": "精确表达，步骤完整，没有任何错误"}),
        ],
    },
    {
        "user_id": "dev_alex",
        "model": "qwen3-4b",
        "messages": [
            ("user", "I need to set up a FastAPI endpoint that accepts file uploads and saves them. Show me the code."),
            ("assistant", "```python\nfrom fastapi import FastAPI, UploadFile, File, HTTPException\nfrom pathlib import Path\nimport shutil\n\napp = FastAPI()\nUPLOAD_DIR = Path('uploads')\nUPLOAD_DIR.mkdir(exist_ok=True)\n\n@app.post('/upload')\nasync def upload_file(file: UploadFile = File(...)):\n    if not file.filename:\n        raise HTTPException(400, 'No file selected')\n    file_path = UPLOAD_DIR / file.filename\n    with file_path.open('wb') as buffer:\n        shutil.copyfileobj(file.file, buffer)\n    return {'filename': file.filename, 'size': file_path.stat().st_size}\n```",
             {"quality_score": 0.80, "feedback": "基本可用，但缺少文件大小限制和安全检查（文件类型过滤）"}),
            ("user", "Add file size limit (10MB) and mime type check (only images)."),
            ("assistant", "```python\nfrom fastapi import FastAPI, UploadFile, File, HTTPException\nfrom pathlib import Path\nimport shutil\nimport mimetypes\n\napp = FastAPI()\nUPLOAD_DIR = Path('uploads')\nUPLOAD_DIR.mkdir(exist_ok=True)\nMAX_SIZE = 10 * 1024 * 1024  # 10MB\nALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}\n\n@app.post('/upload')\nasync def upload_file(file: UploadFile = File(...)):\n    if not file.filename:\n        raise HTTPException(400, 'No file selected')\n    content = await file.read()\n    if len(content) > MAX_SIZE:\n        raise HTTPException(413, f'File too large. Max {MAX_SIZE//1024//1024}MB')\n    mime, _ = mimetypes.guess_type(file.filename)\n    if mime not in ALLOWED_TYPES:\n        raise HTTPException(415, f'File type {mime} not allowed')\n    file_path = UPLOAD_DIR / file.filename\n    file_path.write_bytes(content)\n    return {'filename': file.filename, 'size': len(content)}\n```",
             {"quality_score": 0.90, "feedback": "加了安全检查和限制，但使用async read会把大文件全读入内存"}),
            ("user", "Good, but handle large files with streaming instead."),
            ("assistant", "```python\n@app.post('/upload')\nasync def upload_file(file: UploadFile = File(...)):\n    if not file.filename:\n        raise HTTPException(400, 'No file selected')\n    mime, _ = mimetypes.guess_type(file.filename)\n    if mime not in ALLOWED_TYPES:\n        raise HTTPException(415, f'{mime} not allowed')\n    file_path = UPLOAD_DIR / file.filename\n    total = 0\n    with file_path.open('wb') as buffer:\n        while chunk := await file.read(1024 * 1024):  # 1MB chunks\n            total += len(chunk)\n            if total > MAX_SIZE:\n                buffer.close()\n                file_path.unlink()\n                raise HTTPException(413, f'Exceeds {MAX_SIZE//1024//1024}MB limit')\n            buffer.write(chunk)\n    return {'filename': file.filename, 'size': total}\n```",
             {"quality_score": 0.95, "feedback": "流式处理方案完整，边界条件和安全检查都考虑到了"}),
        ],
    },
    {
        "user_id": "qa_tester",
        "model": "qwen3-4b",
        "messages": [
            ("user", "你是谁？有什么能力？"),
            ("assistant", "我是通义千问(Qwen)系列大语言模型，由阿里云开发。我可以：\n- 回答各类知识问题\n- 编写和调试代码\n- 文本创作和翻译\n- 数学计算和逻辑推理\n- 与用户进行多轮对话\n\n不过我没有实时信息，知识截止到训练数据的时间点。",
             {"quality_score": 0.75, "feedback": "自我介绍了但不够详细，没有说明局限性（如幻觉问题）"}),
            ("user", "如果我让你写一个函数，你的思考过程是怎样的？"),
            ("assistant", "我会这样思考：\n1. 理解需求：明确输入、输出和约束条件\n2. 分析边界：考虑空输入、大数据量、错误情况\n3. 选择算法：权衡时间和空间复杂度\n4. 书写代码：清晰命名、适当注释、遵循语言习惯\n5. 自检：过一遍能否处理正常和异常情况\n\n如果你告诉我具体需求，我可以展示完整的思考+编码过程。",
             {"quality_score": 0.88, "feedback": "回答结构清晰，让用户了解了AI的工作方式"}),
        ],
    },
]


def main():
    store = ConversationStore("data/conversations.db")
    print(f"Seeding {len(DEMO_CONVERSATIONS)} demo conversations...")

    for conv in DEMO_CONVERSATIONS:
        metadata = {}
        # Collect quality scores from assistant messages
        scores = [
            m[2]["quality_score"]
            for m in conv["messages"]
            if m[0] == "assistant" and len(m) > 2 and "quality_score" in m[2]
        ]
        avg_score = sum(scores) / len(scores) if scores else None

        sid = store.create_session(
            user_id=conv["user_id"],
            model=conv["model"],
            metadata={"avg_quality_score": avg_score, "source": "demo_seed"},
        )

        for role, content, *extra in conv["messages"]:
            meta = extra[0] if extra else None
            feedback_text = meta.pop("feedback", None) if meta else None
            quality = meta.pop("quality_score", None) if meta else None

            store.add_message(
                session_id=sid,
                role=role,
                content=content,
                metadata={
                    **(meta or {}),
                    "quality_score": quality,
                    "simulated_feedback": feedback_text,
                },
                stop_reason="stop" if role == "assistant" else None,
            )

    # Print stats
    stats = store.get_statistics()
    print(f"\nImported: {stats['total_sessions']} sessions, {stats['total_messages']} messages")
    print(f"Users: {stats['total_users']}")
    print(f"Role distribution: {stats['role_distribution']}")
    print(f"\nDatabase: {store._db_path}")
    store.close()


if __name__ == "__main__":
    main()
