using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.OCR;
using Emgu.CV.Features2D;

namespace AOCI_7
{
    public partial class Form1 : Form
    {
        private Image<Bgr, byte> sourceImage;
        private Image<Bgr, byte> twistedImg;
        private Image<Bgr, byte> destImage;

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Файлы изображений (*.jpg,  *.jpeg,  *.jpe,  *.jfif,  *.png)  |  *.jpg;  *.jpeg;  *.jpe;  *.jfif; *.png";
            var result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                sourceImage = new Image<Bgr, byte>(fileName);

                imageBox1.Image = sourceImage;
            }

            result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                twistedImg = new Image<Bgr, byte>(fileName);

                imageBox2.Image = twistedImg;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);
            MKeyPoint[] GFP1 = detector.Detect(sourceImage.Convert<Gray, byte>().Mat);

            //создание массива характерных точек исходного изображения (только позиции)
            PointF[] srcPoints = new PointF[GFP1.Length];
            for (int i = 0; i < GFP1.Length; i++)
                srcPoints[i] = GFP1[i].Point;

            PointF[] destPoints; //массив для хранения позиций точек на изменённом изображении
            byte[] status; //статус точек (найдены/не найдены)
            float[] trackErrors; //ошибки
                                 //вычисление позиций характерных точек на новом изображении методом Лукаса-Канаде
            CvInvoke.CalcOpticalFlowPyrLK(
            sourceImage.Convert<Gray, byte>().Mat, //исходное изображение
            twistedImg.Convert<Gray, byte>().Mat,//изменённое изображение
            srcPoints, //массив характерных точек исходного изображения
            new Size(20, 20), //размер окна поиска
            5, //уровни пирамиды
            new MCvTermCriteria(20, 1), //условие остановки вычисления оптического потока
            out destPoints, //позиции характерных точек на новом изображении
            out status, //содержит 1 в элементах, для которых поток был найден
            out trackErrors //содержит ошибки
            );

            //вычисление матрицы гомографии
            Mat homographyMatrix = CvInvoke.FindHomography(destPoints, srcPoints, RobustEstimationAlgorithm.LMEDS);
            var destImage = new Image<Bgr, byte>(sourceImage.Size);
            CvInvoke.WarpPerspective(twistedImg, destImage, homographyMatrix, destImage.Size);
            var output = sourceImage.Clone();
            foreach (PointF p in destPoints)
            {
                CvInvoke.Circle(output, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 2);
            }
            imageBox1.Image = output.Resize(640, 480, Inter.Linear);

            foreach (PointF p in destPoints)
            {
                CvInvoke.Circle(destImage, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 2);
            }
            imageBox2.Image = destImage.Resize(640, 480, Inter.Linear);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);

            var twistedImgGray = twistedImg.Convert<Gray, byte>();
            var baseImgGray = sourceImage.Convert<Gray, byte>();


            //генератор описания ключевых точек
            Brisk descriptor = new Brisk();
            //поскольку в данном случае необходимо посчитать обратное преобразование
            //базой будет являться изменённое изображение
            VectorOfKeyPoint GFP1 = new VectorOfKeyPoint();
            UMat baseDesc = new UMat();
            UMat bimg = twistedImgGray.Mat.GetUMat(AccessType.Read);
            VectorOfKeyPoint GFP2 = new VectorOfKeyPoint();
            UMat twistedDesc = new UMat();
            UMat timg = baseImgGray.Mat.GetUMat(AccessType.Read);
            //получение необработанной информации о характерных точках изображений
            detector.DetectRaw(bimg, GFP1);
            //генерация описания характерных точек изображений
            descriptor.Compute(bimg, GFP1, baseDesc);
            detector.DetectRaw(timg, GFP2); descriptor.Compute(timg, GFP2, twistedDesc);

            //класс позволяющий сравнивать описания наборов ключевых точек
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            //массив для хранения совпадений характерных точек
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            //добавление описания базовых точек
            matcher.Add(baseDesc);
            //сравнение с описанием изменённых
            matcher.KnnMatch(twistedDesc, matches, 2, null);
            //3й параметр - количество ближайших соседей среди которых осуществляется поиск совпадений //4й параметр - маска, в данном случае не нужна

            //маска для определения отбрасываемых значений (аномальных и не уникальных)
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            //определение уникальных совпадений 
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);

            Mat homography;
            //получение матрицы гомографии 
            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(GFP1, GFP2, matches, mask, 2);
            destImage = new Image<Bgr, byte>(sourceImage.Size);
            CvInvoke.WarpPerspective(twistedImg, destImage, homography, destImage.Size);

            Features2DToolbox.DrawMatches(twistedImg, GFP1, sourceImage, GFP2, matches, destImage, new MCvScalar(255, 0, 0), new MCvScalar(255, 0, 0), mask);

            imageBox2.Image = destImage.Resize(640, 480, Inter.Linear);
        }
    }
}
