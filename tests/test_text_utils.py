"""
Test text processing utilities: text_utils.py
"""
import common

from src.rag_service.utils.text_utils import TextProcessor

processor = TextProcessor()


def test_clean_text():
    """Test text cleaning"""
    raw = "  你好！\n\n  Hello world！@#￥%……&*  "
    cleaned = processor.clean_text(raw)
    print("[Text Cleaning]")
    print("Original:", raw)
    print("Cleaned:", cleaned)
    print("-" * 50)


def test_tokenize_zh():
    """Test Chinese tokenization"""
    text = "我喜欢学习人工智能和自然语言处理"
    words = processor.tokenize(text, language="zh")
    print("[Chinese Tokenization]")
    print("Original:", text)
    print("Tokens:", words)
    print("-" * 50)


def test_tokenize_en():
    """Test English tokenization"""
    text = "I love learning artificial intelligence and natural language processing"
    words = processor.tokenize(text, language="en")
    print("[English Tokenization]")
    print("Original:", text)
    print("Tokens:", words)
    print("-" * 50)


def test_chunk_text_zh():
    """Test Chinese text chunking"""
    text = """村口那棵老槐树，已经站了上百年，树干粗壮得要两个成年人合抱，枝丫向四面舒展，夏天时浓荫如盖，是全村人歇凉的好去处。树身上挂着一块褪色的木牌，上面刻着两个小字：守诺。
阿婆住在槐树旁的小平房里，无儿无女，老伴走得早，平日里就靠着打理一小块菜地过日子。她总爱在槐树下坐着，手里捻着针线，目光望向村口的小路，像是在等什么人。
小时候，我常跑去槐树下玩。阿婆从不嫌我吵闹，总会从口袋里摸出一颗水果糖，剥好糖纸塞进我嘴里。甜丝丝的味道，是童年最清晰的记忆。有一回我问她：“阿婆，你天天坐在这儿，是等谁呀？”
阿婆的手顿了顿，眼角泛起温柔的光，轻声说：“等一个老朋友，我们约好，槐花开的时候，就在树下见。”
我那时不懂，只当是阿婆随口一说。直到后来听村里老人讲，才知道那段藏在时光里的故事。
几十年前，阿婆还是个十几岁的姑娘，和邻村的一个青年相爱了。青年家境贫寒，却心地善良，总帮着阿婆家里干活。两人常在槐树下约会，约定等青年从外面挣钱回来，就风风光光娶她过门。
可时局动荡，青年不得不背井离乡，远赴外地谋生。临走那天，也是槐花盛开的季节，两人站在老槐树下，青年紧紧握着阿婆的手说：“等我，不管多久，槐花开时，我一定回来。”
这一等，就是大半辈子。
青年一走便没了音讯，有人说他在外地落了脚，也有人说他遭遇了不测。家人劝阿婆别等了，趁早另寻人家，可阿婆始终不肯。她守着老槐树，守着那句承诺，一年又一年，从青丝等到白发。
每年春天槐花盛开，阿婆都会把树下打扫得干干净净，摆上两个小板凳，一坐就是一整天。风吹过，槐花簌簌落下，像一场温柔的雪，落在她的肩头、发间。
我长大后，离开村子去城里读书，偶尔回乡，依旧能看见阿婆坐在槐树下的身影。她的背更驼了，眼神却依旧执着。有人劝她放下，她只是笑着摇头：“答应了的事，就得守着。说不定哪天，他就回来了。”
去年夏天，我再次回到村子，却没看见槐树下的阿婆。邻居告诉我，阿婆走了，走得很安详，手里还攥着一朵干枯的槐花。
后来，村里人在阿婆的枕头下发现一个小木盒，里面装着一枚磨得光滑的槐花木坠，还有一张泛黄的纸条，上面是青年年轻时的字迹：“待槐花再开，定归赴约。”
老槐树依旧枝繁叶茂，每年春天依旧开满白色的槐花。风一吹，花瓣飘飞，像是在替阿婆，等着那个永远不会赴约的人。
日子一天天过去，村里的孩子依旧在槐树下奔跑嬉戏，只是再也没有人坐在树下，一等就是一生。可那棵老槐树，却像一座沉默的丰碑，守着一段跨越岁月的深情，告诉每一个路过的人：世间最珍贵的，从来不是轰轰烈烈的誓言，而是平平淡淡的坚守。"""

    chunks = processor.chunk_text(text, language="zh")
    print("[Chinese Text Chunking]")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:", chunk)
    print("-" * 50)


def test_chunk_text_en():
    """Test English text chunking"""
    text = """
    The old locust tree at the village entrance has stood there for over a century. Its trunk is so thick that two adults need to stretch their arms to wrap around it, with branches spreading in all directions. In summer, its thick shade covers a wide area, making it a perfect resting spot for all villagers. A faded wooden sign hangs on the tree, carved with two small characters: Keep the Promise.
Grandma lived in a small cottage beside the locust tree. She had no children, and her husband passed away early. She made a living by tending a small vegetable patch and would often sit under the tree, sewing, her eyes fixed on the path leading to the village, as if waiting for someone.
When I was little, I often ran to play under the tree. Grandma never minded my noise; she would always take a fruit candy from her pocket, peel off the wrapper, and put it in my mouth. The sweet taste remains my clearest childhood memory. Once I asked her, “Grandma, who are you waiting for here every day?”
Grandma’s hands paused for a moment, warmth shining in her eyes. She whispered, “An old friend. We promised to meet under this tree when the locust flowers bloom.”
I didn’t understand back then and thought she was just speaking casually. Only later, when I heard the elders in the village talk, did I learn the story hidden in time.
Decades ago, Grandma was a teenage girl in love with a young man from a neighboring village. The young man was poor but kind-hearted, always helping her family with chores. They often dated under the locust tree, promising that he would marry her in a grand ceremony once he earned enough money away from home.
But turbulent times forced him to leave and make a living far away. On the day he departed, it was also the season of blooming locust flowers. Standing under the old tree, he held her hands tightly and said, “Wait for me. No matter how long it takes, I will come back when the locust flowers bloom.”
That wait lasted most of her life.
After he left, there was no news of him. Some said he had settled elsewhere; others claimed he had met with misfortune. Her family urged her to give up and find someone else, but Grandma refused. She kept waiting by the locust tree, holding onto that promise, year after year, from black hair to white.
Every spring when the locust flowers bloomed, Grandma would clean the area under the tree neatly, place two small benches, and sit there all day. The wind blew, and the petals fell softly like gentle snow, covering her shoulders and hair.
When I grew up and left the village to study in the city, I still saw Grandma sitting under the tree whenever I returned. Her back became more hunched, but her gaze remained persistent. People tried to persuade her to let go, but she only shook her head with a smile. “A promise is a promise. Maybe one day he will come back.”
Last summer, I returned to the village again, but Grandma was no longer under the locust tree. My neighbor told me she had passed away peacefully, holding a dried locust flower in her hand.
Later, the villagers found a small wooden box under her pillow. Inside was a smooth locust wood pendant and a yellowed note with the young man’s handwriting from his youth: “When the locust flowers bloom again, I will return to keep my promise.”
The old locust tree still grows lush and green, still covered in white blossoms every spring. When the wind blows, the petals flutter, as if waiting for the one who would never arrive, on her behalf.
Days pass by. Village children still run and play under the tree, but no one sits there waiting for a lifetime anymore. Yet the old locust tree stands like a silent monument, guarding a deep affection that crossed the years, telling everyone who passes by: The most precious thing in the world is never a grand vow, but a quiet, steady perseverance.
    """
    chunks = processor.chunk_text(text, language="en")
    print("[English Text Chunking]")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:", chunk)
    print("-" * 50)


if __name__ == "__main__":
    test_clean_text()
    test_tokenize_zh()
    test_tokenize_en()
    test_chunk_text_zh()
    test_chunk_text_en()
    print("All tests completed!")