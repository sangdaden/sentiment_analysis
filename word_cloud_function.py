import matplotlib.pyplot as plt
from wordcloud import WordCloud

positive_words = [
    "ok", "thoải mái", "hiệu quả", "nâng tông", "mướt", "mịn", "dễ chịu", "mượt", "thơm", "dịu nhẹ", "tươi trẻ", "mềm mại", "làm sáng", "làm đều màu",
    "chống lão hóa", "dưỡng ẩm", "ngừa mụn", "thư giãn", "không kích ứng", "tăng độ đàn hồi",
    "sạch sâu", "giảm thâm", "mát lạnh", "làm dịu", "tươi mát", "phục hồi", "dưỡng trắng",
    "trẻ hóa", "tẩy tế bào chết", "sáng da", "khôi phục", "bảo vệ", "chống nắng",

    "xem review", "ưu tiên thương hiệu", "chọn sản phẩm tự nhiên", "mua số lượng lớn", "canh giảm giá", "tìm khuyến mãi", "tin tưởng", "tặng quà", 
    "đăng ký thành viên", "quay lại mua", "đặt hàng online", "dùng voucher", 
    "chọn sản phẩm chính hãng",
    
    "thích", "tốt", "xuất sắc", "tuyệt vời", "ổn",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "nhanh",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "thân thiện",
    "cao cấp", "độc đáo", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "thúc đẩy", "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "phù hợp", "tận tâm", "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
]

negative_words = [
    "mắc", "khô", "nặng", "dầu", "kích ứng", "dị ứng", "tắc nghẽn lỗ chân lông", 
    "ngứa", "bóng nhờn", "tạo cảm giác dính", "khó thấm", "bít tắc", 
    "nặng mặt", "kém hiệu quả", "bóng dầu", "thô ráp", "lão hóa", "đỏ rát",
    "mùi hôi", "không hiệu quả",

    "chọn sản phẩm kém chất lượng", "chọn sản phẩm không phù hợp",

    "kém", "tệ", "buồn", "chán", "không dễ chịu", "không chất lượng"
    "kém chất lượng", "không thích", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó chịu", "gây khó dễ", "rườm rà", "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền",
    'không hài lòng', 'không đáng', 'quá tệ', 'rất tệ',
    'thất vọng', 'chán', 'tệ hại', 'kinh khủng', 'không ưng ý'
]


def analyze_product_reviews(df, product_id):
    """
    Phân tích và tạo WordCloud cho một mã sản phẩm cụ thể.

    Parameters:
        df (DataFrame): DataFrame chứa dữ liệu sản phẩm và nhận xét.
        product_id (int): Mã sản phẩm cần phân tích.

    Returns:
        None
    """
    # Lọc dữ liệu
    product_data = df[df["ma_san_pham"] == product_id]
    if product_data.empty:
        print(f"Không tìm thấy dữ liệu cho mã sản phẩm: {product_id}")
        return

    # Tính số lượng nhận xét
    positive_reviews = product_data[product_data["sentiment"] == "positive"]
    negative_reviews = product_data[product_data["sentiment"] == "negative"]

    print(f"Số nhận xét tích cực: {len(positive_reviews)}")
    print(f"Số nhận xét tiêu cực: {len(negative_reviews)}")

    # Hàm tạo WordCloud
    def generate_wordcloud(text, title):
        text = text.dropna().astype(str)
        if text.empty:
            print(f"Không có dữ liệu để tạo WordCloud cho {title}")
            return
        wordcloud = WordCloud(max_words=50, width=800, height=400, background_color="white").generate(" ".join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.show()

    # Tạo WordCloud
    generate_wordcloud(positive_reviews["noi_dung_binh_luan_sau_xu_ly"], "Positive Reviews")
    generate_wordcloud(negative_reviews["noi_dung_binh_luan_sau_xu_ly"], "Negative Reviews")