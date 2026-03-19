"""Generate sample JSONL data for testing the pipeline."""

import json
import os
import random

SAMPLES = {
    "bao_chi": [
        "Theo tin từ Bộ Y tế, tình hình dịch bệnh đang được kiểm soát tốt trên toàn quốc. Các cơ sở y tế đã sẵn sàng ứng phó với mọi tình huống phát sinh.",
        "Giải bóng đá V-League 2024 đã chính thức khởi tranh với sự tham gia của 14 câu lạc bộ. Trận khai mạc diễn ra tại sân vận động Mỹ Đình thu hút hàng nghìn khán giả.",
        "Thị trường bất động sản quý 3 ghi nhận nhiều tín hiệu tích cực khi nguồn cung căn hộ mới tăng mạnh tại các thành phố lớn như Hà Nội và TP.HCM.",
        "Vụ tai nạn giao thông nghiêm trọng xảy ra trên quốc lộ 1A làm 3 người tử vong và 5 người bị thương. Cơ quan chức năng đang điều tra nguyên nhân.",
        "Tổng thống Mỹ đã có cuộc hội đàm với Thủ tướng Việt Nam nhân chuyến thăm chính thức. Hai bên đã ký kết nhiều văn kiện hợp tác quan trọng.",
    ],
    "hanh_chinh": [
        "Căn cứ Luật Tổ chức chính quyền địa phương năm 2015, Ủy ban nhân dân tỉnh ban hành Quyết định về việc phê duyệt quy hoạch sử dụng đất giai đoạn 2021-2030.",
        "Kính gửi: Sở Giáo dục và Đào tạo. Thực hiện Công văn số 1234/BGDĐT-GDTrH ngày 15/3/2024 của Bộ Giáo dục và Đào tạo về việc hướng dẫn tuyển sinh lớp 10.",
        "Điều 1. Phạm vi điều chỉnh. Nghị định này quy định chi tiết thi hành một số điều của Luật Bảo vệ môi trường về đánh giá tác động môi trường.",
        "Theo quy định tại Thông tư số 08/2023/TT-BTC, các tổ chức tín dụng phải thực hiện báo cáo tài chính theo mẫu biểu quy định tại Phụ lục đính kèm.",
        "Ủy ban nhân dân xã quyết định: Cấp giấy chứng nhận quyền sử dụng đất cho hộ gia đình ông Nguyễn Văn A, địa chỉ thường trú tại thôn B, xã C.",
    ],
    "khoa_hoc": [
        "Nghiên cứu này đề xuất một phương pháp mới trong xử lý ngôn ngữ tự nhiên tiếng Việt sử dụng mô hình Transformer. Kết quả thực nghiệm cho thấy độ chính xác đạt 95.2% trên tập dữ liệu benchmark.",
        "Quá trình quang hợp ở thực vật xảy ra chủ yếu tại lục lạp, nơi năng lượng ánh sáng được chuyển hóa thành năng lượng hóa học dưới dạng glucose và các hợp chất hữu cơ khác.",
        "Theo kết quả phân tích thống kê với mức ý nghĩa p < 0.05, có sự khác biệt có ý nghĩa thống kê giữa nhóm thực nghiệm và nhóm đối chứng về chỉ số BMI.",
        "Vật liệu nano carbon đang được nghiên cứu ứng dụng rộng rãi trong lĩnh vực y sinh học, đặc biệt trong việc phát triển hệ thống dẫn truyền thuốc nhắm mục tiêu.",
        "Thuật toán học sâu CNN được áp dụng để phân loại hình ảnh y khoa với kiến trúc ResNet-50 đã được tinh chỉnh trên tập dữ liệu gồm 10.000 ảnh X-quang phổi.",
    ],
    "chinh_luan": [
        "Trong bối cảnh toàn cầu hóa và hội nhập quốc tế sâu rộng, việc bảo vệ chủ quyền quốc gia trên không gian mạng đang trở thành một vấn đề cấp bách.",
        "Đảng ta luôn khẳng định giáo dục là quốc sách hàng đầu. Đầu tư cho giáo dục chính là đầu tư cho tương lai của đất nước và dân tộc.",
        "Cuộc cách mạng công nghiệp 4.0 đặt ra nhiều thách thức nhưng cũng mở ra cơ hội to lớn cho Việt Nam trong việc đẩy nhanh quá trình công nghiệp hóa, hiện đại hóa.",
        "Tham nhũng là kẻ thù nội xâm, là mối nguy hại lớn đối với sự tồn vong của chế độ. Công tác phòng chống tham nhũng cần được đẩy mạnh hơn nữa.",
        "Xây dựng nhà nước pháp quyền xã hội chủ nghĩa là nhiệm vụ trọng tâm trong giai đoạn hiện nay, đòi hỏi sự nỗ lực của toàn Đảng, toàn dân.",
    ],
    "sinh_hoat": [
        "Hôm nay trời đẹp quá, mình đưa con đi công viên chơi. Bé thích thú với mấy con vịt bơi dưới hồ, đòi cho vịt ăn mãi không chịu về.",
        "Chị ơi cho em hỏi quán phở ngon gần đây ở đâu ạ? Em mới chuyển tới khu này nên chưa biết chỗ nào ngon. Cảm ơn chị!",
        "Cuối tuần rồi cả nhà mình nấu lẩu, vui ghê. Ông bà nội cũng qua chơi, mấy đứa nhỏ được ông bà cho quà mừng hết lớn.",
        "Sáng nay đi chợ mua được mớ rau muống tươi ngon mà rẻ lắm, chỉ có 5 nghìn một bó. Về nhà xào tỏi ăn cơm ngon lành.",
        "Mấy bạn ơi, ai biết cách trồng hoa hồng trên sân thượng không? Mình mới mua mấy chậu mà không biết chăm sóc thế nào cho hoa nở đẹp.",
    ],
}


def generate_sample_data(output_path: str, samples_per_label: int = 5):
    """Generate sample JSONL data."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    records = []
    for label, texts in SAMPLES.items():
        for text in texts[:samples_per_label]:
            records.append({"text": text, "label": label})

    random.shuffle(records)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} sample records to {output_path}")


if __name__ == "__main__":
    generate_sample_data("data/raw/sample.jsonl")
