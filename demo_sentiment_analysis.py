import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Read data
data = pd.read_csv("Danh_gia_final.csv", encoding='utf-8')
data['sentiment'] = data['so_sao'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

data_san_pham = pd.read_csv("San_pham.csv", encoding='utf-8')

#--------------
# GUI
# Giao diện Streamlit với Tabs
st.image('hasaki_banner_2.jpg')
st.title("🧺 Hasaki Sentiment Analysis😊")
# st.write("Chọn chế độ gợi ý sản phẩm phù hợp!")

menu = st.sidebar.selectbox(
    "🌟 **Menu Chức năng**",
    ["Đặt Vấn Đề", "Thực Hiện Dự Án", "Xây Dựng Mô Hình", "Dự Đoán Cảm Xúc", "Các Chức Năng Khác"]
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
    # Đọc model
    with open('Logistic Regression.pkl', 'rb') as file:  
        svm_model = pickle.load(file)
    # doc model count len
    with open('tfidf_vectorizer.pkl', 'rb') as file:  
        count_model = pickle.load(file)

    st.subheader("Chọn dữ liệu")
    flag = False
    lines = None

    type = st.radio("Bạn muốn tải lên hay nhập vào?", options=("Tải lên", "Nhập"))
    if type=="Tải lên":
        # Hướng dẫn tải tệp
        st.write("📄 **Hướng dẫn tải tệp:**")
        st.write("- Định dạng tệp: `.txt` hoặc `.csv`.")
        st.write("- Mỗi dòng trong tệp tương ứng với một bình luận, ví dụ:")
        st.code("Dịch vụ rất tốt\nThái độ nhân viên không chuyên nghiệp\nChất lượng sản phẩm ổn")

        # Upload file
        uploaded_file_1 = st.file_uploader("Chọn file (csv hoặc txt)", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0].astype(str)                          
    elif type=="Nhập":
        # Hướng dẫn nhập nội dung
        st.write("✏️ **Hướng dẫn nhập nội dung:**")
        st.write("- Nhập từng bình luận trên mỗi dòng.")
        st.write("- Ví dụ:")
        st.code("Dịch vụ rất tốt\nThái độ nhân viên không chuyên nghiệp\nChất lượng sản phẩm ổn")

        # Nhập nội dung
        content = st.text_area(label="Nhập nội dung:")
        if content.strip():
            lines = np.array(content.splitlines())  # Tách từng dòng

    # Nút dự đoán
    if st.button("Dự đoán"):
        if lines is not None and len(lines) > 0:
            # Thực hiện dự đoán
            st.write("🔍 **Kết quả Dự đoán:**")
            x_new = count_model.transform(lines)        
            y_pred_new = svm_model.predict(x_new)
            
            # Hiển thị kết quả bằng thẻ cảm xúc
            for i, line in enumerate(lines):
                emotion = y_pred_new[i]
                if emotion == "positive":
                    st.success(f"**Bình luận:** {line}\n\n**Cảm xúc:** 😄 Tích cực")
                elif emotion == "negative":
                    st.error(f"**Bình luận:** {line}\n\n**Cảm xúc:** 😡 Tiêu cực")
                else:
                    st.info(f"**Bình luận:** {line}\n\n**Cảm xúc:** 😐 Trung lập")
        else:
            st.warning("Vui lòng nhập dữ liệu hoặc tải lên file để dự đoán!")

elif menu == 'Xây Dựng Mô Hình':
    st.subheader("Xây dựng mô hình")
    st.write("##### 1. Dữ liệu")
    st.dataframe(data[['noi_dung_binh_luan', 'noi_dung_binh_luan_sau_xu_ly', 'so_sao']].head(3))
    st.dataframe(data[['noi_dung_binh_luan', 'noi_dung_binh_luan_sau_xu_ly', 'so_sao']].tail(3))

    st.write("##### 2. Thống kê theo số sao và các nhãn")
    st.write("###### ⭐ Theo số sao")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x='so_sao', palette='coolwarm', ax=ax)
    ax.set_title("Phân bố số sao", fontsize=14)
    ax.set_xlabel("Số sao", fontsize=12)
    ax.set_ylabel("Số lượng", fontsize=12)
    st.pyplot(fig)

    sentiment_counts = data['sentiment'].value_counts()
    st.write("###### 😊 Theo cảm xúc")
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.coolwarm([0.2, 0.5, 0.8]) 
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    ax.set_title("Phân bổ Sentiment")
    st.pyplot(fig)

    st.write("##### 3. Đánh giá mô hình sử dụng")
    st.write("###### 💻 Confusion Matrix")
    st.image('confusion_matrix.png')

    st.write("###### 🎨 ROC Curve")
    st.image('roc_curve.png')

    st.write("##### 4. Tổng quan")
    st.write("=> Chọn mô hình **Logistic Regression** khi cho kết quả dự đoán tốt.")

elif menu == "Các Chức Năng Khác":
    tab1, tab2, tab3 = st.tabs(["🔍 Phân tích sản phẩm", "🙆‍♂️ Khách hàng nổi bật", "⌚ So sánh cảm xúc theo thời gian"])
    with tab1:
        st.subheader("Phân tích WordCloud theo sản phẩm")

        # Kết hợp dữ liệu từ hai bảng
        merged_data = data.merge(data_san_pham, on="ma_san_pham", how="inner")

        # Chọn mã sản phẩm
        product_names  = merged_data['ten_san_pham'].unique()
        selected_product_name = st.selectbox("Chọn sản phẩm:", product_names)

        # Lấy mã sản phẩm tương ứng với tên sản phẩm đã chọn
        selected_product_id = merged_data[merged_data['ten_san_pham'] == selected_product_name]['ma_san_pham'].iloc[0]

        # Lọc dữ liệu theo mã sản phẩm
        product_data = merged_data[merged_data['ma_san_pham'] == selected_product_id]

        if len(product_data) > 0:
            # Phân loại tích cực/tiêu cực
            positive_reviews = product_data[product_data['so_sao'] >= 4]['noi_dung_binh_luan_sau_xu_ly']
            negative_reviews = product_data[product_data['so_sao'] <= 3]['noi_dung_binh_luan_sau_xu_ly']

            # Tạo WordCloud
            positive_text = " ".join(positive_reviews.astype(str))
            negative_text = " ".join(negative_reviews.astype(str))

            if positive_text.strip():
                wordcloud_positive = WordCloud(max_words=50, width=800, height=400, background_color="white").generate(positive_text)
                st.write(f"Tổng số bình luận tích cực: {len(positive_reviews)}")
                st.write("### WordCloud Tích Cực")
                st.image(wordcloud_positive.to_array())
            else:
                st.write("Không có dữ liệu đủ để tạo WordCloud Tích Cực.")

            if negative_text.strip():
                wordcloud_negative = WordCloud(max_words=50, width=800, height=400, background_color="black").generate(negative_text)
                st.write(f"Tổng số bình luận tiêu cực: {len(negative_reviews)}")
                st.write("### WordCloud Tiêu Cực")
                st.image(wordcloud_negative.to_array())
            else:
                st.write("Không có dữ liệu đủ để tạo WordCloud Tiêu Cực.")
        else:
            st.write("Không có dữ liệu cho sản phẩm này.")


    with tab2:
        # Tính năng: Phân tích khách hàng nổi bật
        st.write("### Phân tích khách hàng nổi bật")
        
        # Tính số lượng bình luận của mỗi khách hàng
        customer_comments = data.groupby('ma_khach_hang')['noi_dung_binh_luan'].count().sort_values(ascending=False)
        
        # Hiển thị 10 khách hàng bình luận nhiều nhất
        top_customers = customer_comments.head(10)
        
        # Biểu đồ cột hiển thị khách hàng bình luận nhiều nhất
        fig, ax = plt.subplots(figsize=(10, 6))
        top_customers.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('10 khách hàng bình luận nhiều nhất', fontsize=16)
        ax.set_xlabel('Mã khách hàng', fontsize=12)
        ax.tick_params(axis='x', rotation=45) 
        ax.set_ylabel('Số bình luận', fontsize=12)
        st.pyplot(fig)

        # Phân tích khách hàng có số sao cực kỳ cao hoặc thấp
        top_positive_customers = data[data['so_sao'] == 5].groupby('ma_khach_hang')['noi_dung_binh_luan'].count().sort_values(ascending=False).head(10)
        top_negative_customers = data[data['so_sao'] == 1].groupby('ma_khach_hang')['noi_dung_binh_luan'].count().sort_values(ascending=False).head(10)

        # Tạo bảng cho khách hàng tích cực (5 sao)
        st.write("### Khách hàng có số sao cực kỳ cao (Tích cực - 5 sao):")
        st.dataframe(top_positive_customers)

        # Biểu đồ cột cho khách hàng tích cực
        fig, ax = plt.subplots(figsize=(10, 6))
        top_positive_customers.plot(kind='bar', color='green', ax=ax)
        ax.set_title('Khách hàng Tích Cực (5 sao)', fontsize=16)
        ax.set_xlabel('Mã khách hàng', fontsize=12)
        ax.tick_params(axis='x', rotation=45) 
        ax.set_ylabel('Số bình luận', fontsize=12)
        st.pyplot(fig)

        # Tạo bảng cho khách hàng tiêu cực (1 sao)
        st.write("### Khách hàng có số sao cực kỳ thấp (Tiêu cực - 1 sao):")
        st.dataframe(top_negative_customers)

        # Biểu đồ cột cho khách hàng tiêu cực
        fig, ax = plt.subplots(figsize=(10, 6))
        top_negative_customers.plot(kind='bar', color='red', ax=ax)
        ax.set_title('Khách hàng Tiêu Cực (1 sao)', fontsize=16)
        ax.set_xlabel('Mã khách hàng', fontsize=12)
        ax.tick_params(axis='x', rotation=45) 
        ax.set_ylabel('Số bình luận', fontsize=12)
        st.pyplot(fig)

    with tab3:
        st.write("### So sánh cảm xúc theo thời gian")
    
        # Chuyển đổi cột 'ngay_binh_luan' sang datetime, thay thế các giá trị không hợp lệ bằng NaT
        data['ngay_binh_luan'] = pd.to_datetime(data['ngay_binh_luan'], errors='coerce')
        
        # Loại bỏ các dòng có giá trị NaT (ngày không hợp lệ)
        data = data.dropna(subset=['ngay_binh_luan'])
        
        # Phân loại cảm xúc theo số sao
        data['cam_xuc'] = data['so_sao'].apply(lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 3 else 'Neutral'))
        
        # Thêm cột tháng/năm để phân tích theo thời gian
        data['thang_nam'] = data['ngay_binh_luan'].dt.to_period('M')

        # Tính số lượng các cảm xúc theo tháng
        emotion_count_by_month = data.groupby(['thang_nam', 'cam_xuc']).size().unstack(fill_value=0)
        
        # Tạo biểu đồ so sánh cảm xúc theo thời gian
        fig, ax = plt.subplots(figsize=(10, 6))
        emotion_count_by_month.plot(kind='line', marker='o', ax=ax, color=['green', 'red', 'blue'])
        
        ax.set_title('So sánh cảm xúc theo thời gian', fontsize=16)
        ax.set_xlabel('Thời gian (Tháng/Năm)', fontsize=12)
        ax.set_ylabel('Số lượng bình luận', fontsize=12)
        ax.set_xticklabels(emotion_count_by_month.index.astype(str), rotation=45)
        st.pyplot(fig)
        
        # Hiển thị bảng số liệu
        st.write("### Số lượng bình luận phân theo cảm xúc theo tháng")
        st.dataframe(emotion_count_by_month)
