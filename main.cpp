#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char** argv) {

    std::vector<std::string> etiketler;
    std::ifstream dosya("labels.txt");
    std::string satir;
    while (std::getline(dosya, satir)) {
        etiketler.push_back(satir);
    }

    cv::dnn::Net model = cv::dnn::readNetFromONNX("best.onnx");
    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture kamera("tank.mp4");
    if (!kamera.isOpened()) {
        printf("Video acilamadi!\n");
        return -1;
    }

    cv::Mat frame;

    while (true) {
        kamera >> frame;
        if (frame.empty()) break;

        int gercekGenislik  = frame.cols;
        int gercekYukseklik = frame.rows;

        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0 / 255.0,
            cv::Size(640, 640),
            cv::Scalar(), true, false
        );

        model.setInput(blob);
        cv::Mat cikti = model.forward();

        cv::Mat satirlar = cikti.reshape(1, cikti.size[1]);
        cv::transpose(satirlar, satirlar);

        std::vector<cv::Rect> kutular;
        std::vector<float>    guvenler;
        std::vector<int>      classlar;

        float esik = 0.5;

        for (int i = 0; i < satirlar.rows; i++) {
            cv::Mat skorlar = satirlar.row(i).colRange(4, satirlar.cols);
            cv::Point enYuksekYer;
            double enYuksekDeger;
            cv::minMaxLoc(skorlar, nullptr, &enYuksekDeger, nullptr, &enYuksekYer);

            if (enYuksekDeger < esik) continue;

            float cx = satirlar.at<float>(i, 0);
            float cy = satirlar.at<float>(i, 1);
            float bw = satirlar.at<float>(i, 2);
            float bh = satirlar.at<float>(i, 3);

            int x = (int)((cx - bw / 2) * gercekGenislik  / 640);
            int y = (int)((cy - bh / 2) * gercekYukseklik / 640);
            int w = (int)(bw * gercekGenislik  / 640);
            int h = (int)(bh * gercekYukseklik / 640);

            kutular.push_back(cv::Rect(x, y, w, h));
            guvenler.push_back((float)enYuksekDeger);
            classlar.push_back(enYuksekYer.x);
        }

        std::vector<int> kalanlar;
        cv::dnn::NMSBoxes(kutular, guvenler, esik, 0.4, kalanlar);

        for (int idx : kalanlar) {
            cv::rectangle(frame, kutular[idx], cv::Scalar(0, 255, 0), 2);
            std::string yazi = etiketler[classlar[idx]] +
                               " %" + std::to_string((int)(guvenler[idx] * 100));
            cv::putText(frame, yazi,
                cv::Point(kutular[idx].x, kutular[idx].y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Askeri Hedef Tespiti", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}