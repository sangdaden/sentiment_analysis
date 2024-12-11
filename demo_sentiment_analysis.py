import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
from wordcloud import WordCloud

# 1. Read data
data = pd.read_csv("Danh_gia_final.csv", encoding='utf-8')

# 2. Data pre-processing


#--------------
# GUI
# Giao diện Streamlit với Tabs
st.image('hasaki_banner_2.jpg')
st.title("🧺 Hasaki Sentiment Analysis😊😞😶")
# st.write("Chọn chế độ gợi ý sản phẩm phù hợp!")

menu = st.sidebar.selectbox(
    "🌟 **Menu Chức năng**",
    ["Đặt Vấn Đề", "Thực Hiện Dự Án", "Dự Đoán Cảm Xúc", "Bảng Tổng Hợp Thông Minh"]
    #  "Phân tích khách hàng nổi bật", "Từ khóa nổi bật theo cảm xúc", 
    #  "So sánh cảm xúc theo thời gian", "Dữ liệu mẫu", "Phân tích sản phẩm",
    #  "Business Overview", "Miêu tả cách thực hiện", , "Trực quan hóa dữ liệu"]
)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Thanh Sang & 
                 Tạ Quang Hưng""")
st.sidebar.write("""#### Giảng viên hướng dẫn:
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện:
                 12/2024""")

if menu == 'Đặt Vấn Đề':
    # Phần mở đầu
    st.write("""
        💬 **Khách hàng nói gì về sản phẩm?**  
        Trong thời đại số, khách hàng để lại rất nhiều nhận xét và đánh giá trên các nền tảng trực tuyến.  
        Hiểu được cảm xúc từ những phản hồi này giúp doanh nghiệp cải thiện sản phẩm và dịch vụ.
    """)

    # Các vấn đề gặp phải
    st.subheader("🚩 Vấn đề cần giải quyết")
    st.write("""
        1. **Dữ liệu lớn**: Rất khó phân tích hàng ngàn nhận xét thủ công.  
        2. **Cảm xúc đa dạng**: Đánh giá có thể tích cực, tiêu cực hoặc trung tính.  
        3. **Ngữ cảnh phức tạp**: Một số nhận xét có ẩn ý hoặc mỉa mai.  
    """)

    # Mục tiêu hệ thống
    st.subheader("🎯 Mục tiêu")
    st.write("""
        1. **Tự động phân loại cảm xúc**: Xác định nhận xét tích cực, tiêu cực, hoặc trung tính.  
        2. **Cải thiện sản phẩm**: Dựa trên phản hồi tiêu cực để nâng cao chất lượng.  
        3. **Tăng sự hài lòng**: Tối ưu hóa trải nghiệm khách hàng và chiến lược kinh doanh.  
    """)

    # Kết luận
    st.write("""
        Hệ thống **Sentiment Analysis** là công cụ mạnh mẽ để doanh nghiệp hiểu sâu hơn về khách hàng,  
        đưa ra các quyết định chính xác và phát triển bền vững.
    """)

elif menu == 'Thực Hiện Dự Án':
    # Mở đầu
    st.write("""
    🛠️ **Sentiment Analysis là gì?**  
    Sentiment Analysis (Phân tích cảm xúc) là quá trình phân loại phản hồi của khách hàng thành các nhóm cảm xúc:  tích cực, tiêu cực, hoặc trung tính. Đây là công cụ quan trọng để hiểu khách hàng và cải thiện sản phẩm.
    """)

    # Quy trình thực hiện
    st.subheader("⚙️ Quy trình thực hiện")
    st.write("""
    1. **Thu thập dữ liệu**:  
    - Lấy dữ liệu đánh giá từ các nguồn như website, mạng xã hội, hoặc email phản hồi.  

    2. **Tiền xử lý dữ liệu**:  
    - Làm sạch văn bản: xóa ký tự đặc biệt, chuyển về chữ thường.  
    - Loại bỏ stopwords và sử dụng stemming/lemmatization để chuẩn hóa.  

    3. **Xây dựng mô hình phân loại**:  
    - Sử dụng các phương pháp phổ biến:  
        - **Truyền thống**: Naive Bayes, SVM, Logistic Regression.  
        - **Hiện đại**: Mô hình dựa trên deep learning như LSTM hoặc transformer (BERT).  

    4. **Đánh giá hiệu quả**:  
    - Sử dụng các chỉ số như Accuracy, Precision, Recall, F1-score.  

    5. **Ứng dụng thực tế**:  
    - Gắn nhãn cảm xúc cho đánh giá mới.  
    - Tổng hợp thống kê cảm xúc theo sản phẩm để ra quyết định kinh doanh.
    """)

    # Ưu điểm & Hạn chế
    st.subheader("📊 Ưu điểm và Hạn chế")
    st.write("""
    - **Ưu điểm**:  
    ✅ Hiểu rõ phản hồi khách hàng.  
    ✅ Tự động hóa phân tích dữ liệu lớn.  
    ✅ Phát hiện sớm vấn đề tiềm ẩn từ đánh giá tiêu cực.  

    - **Hạn chế**:  
    ⚠️ Độ chính xác giảm nếu dữ liệu không đầy đủ hoặc không đồng nhất.  
    ⚠️ Khó phân tích các ngữ cảnh phức tạp (ẩn ý, mỉa mai).  
    """)

    # Kết luận
    st.subheader("📌 Kết luận")
    st.write("""
    Bằng cách sử dụng Sentiment Analysis, doanh nghiệp có thể tối ưu hóa sản phẩm, cải thiện dịch vụ,  
    và tăng mức độ hài lòng của khách hàng một cách bền vững.
    """)

elif menu == 'Dự Đoán Cảm Xúc':
    #6. Load models 
    # Đọc model
    # import pickle
    with open('Logistic Regression.pkl', 'rb') as file:  
        svm_model = pickle.load(file)
    # doc model count len
    with open('tfidf_vectorizer.pkl', 'rb') as file:  
        count_model = pickle.load(file)

    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True

    if flag:
        st.write("Nội dung:")
        if len(lines)>0:
            st.code(lines)        
            x_new = count_model.transform(lines)        
            y_pred_new = svm_model.predict(x_new)       
            st.code("Dự đoán mới (Positive, Neutral, Negative): " + str(y_pred_new)) 

if menu == "Bảng Tổng Hợp Thông Minh":
    st.subheader("Phân tích WordCloud theo sản phẩm")
    
    # Chọn mã sản phẩm
    product_ids = data['ma_san_pham'].unique()
    selected_product = st.selectbox("Chọn sản phẩm:", product_ids)

    # Lọc dữ liệu theo sản phẩm
    product_data = data[data['ma_san_pham'] == selected_product]

    if len(product_data) > 0:
        # Phân loại tích cực/tiêu cực
        positive_reviews = product_data[product_data['so_sao'] >= 4]['noi_dung_binh_luan_sau_xu_ly']
        negative_reviews = product_data[product_data['so_sao'] <= 3]['noi_dung_binh_luan_sau_xu_ly']

        # Tạo WordCloud
        positive_text = " ".join(positive_reviews.astype(str))
        negative_text = " ".join(negative_reviews.astype(str))

        wordcloud_positive = WordCloud(max_words=50, width=800, height=400, background_color="white").generate(positive_text)
        wordcloud_negative = WordCloud(max_words=50, width=800, height=400, background_color="black").generate(negative_text)

        # Hiển thị tổng số bình luận
        st.write(f"Tổng số bình luận tích cực: {len(positive_reviews)}")
        st.write(f"Tổng số bình luận tiêu cực: {len(negative_reviews)}")

        # Hiển thị WordCloud
        st.write("### WordCloud Tích Cực")
        st.image(wordcloud_positive.to_array())

        st.write("### WordCloud Tiêu Cực")
        st.image(wordcloud_negative.to_array())
    else:
        st.write("Không có dữ liệu cho sản phẩm này.")