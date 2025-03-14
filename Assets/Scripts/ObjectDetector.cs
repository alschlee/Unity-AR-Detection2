using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Unity.Barracuda;
using UnityEngine.XR.ARSubsystems;
using System.Linq;

public class ObjectDetector : MonoBehaviour
{
    public NNModel modelAsset;
    public TextAsset labelsAsset;
    public int imageSize = 640;
    public float confidenceThreshold = 0.5f;

    private ARCameraManager arCameraManager;
    private Model m_RuntimeModel;
    private IWorker m_Worker;
    private string[] m_Labels;
    private Texture2D m_Input;

    void Awake()
    {
        arCameraManager = GetComponentInChildren<ARCameraManager>();
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived += OnCameraFrameReceived;
        }
        else
        {
            Debug.LogError("ARCameraManager 컴포넌트를 찾을 수 없습니다.");
        }
    }

    void Start()
    {
        // 모델 및 라벨 로드
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, m_RuntimeModel);
        m_Labels = labelsAsset.text.Split('\n');

        // 입력 텍스처 초기화
        m_Input = new Texture2D(imageSize, imageSize, TextureFormat.RGB24, false);
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        if (eventArgs.textures.Count == 0)
            return;

        // 카메라 텍스처 가져오기
        Texture2D cameraTexture = eventArgs.textures[0];

        // 입력 텐서 생성
        using (var tensor = TransformInput(cameraTexture))
        {
            // 추론 실행
            m_Worker.Execute(tensor);

            // 출력 텐서 가져오기
            var output = m_Worker.PeekOutput();

            // 결과 처리
            ProcessOutput(output);
        }
    }

    private Tensor TransformInput(Texture2D texture)
    {
        // 이미지를 모델 입력 형식으로 변환
        float[] inputData = new float[imageSize * imageSize * 3];
        Color[] pixels = texture.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // 모델의 정규화 방식에 맞게 처리 (여기서는 0~1로 정규화)
            inputData[i * 3] = pixels[i].r;
            inputData[i * 3 + 1] = pixels[i].g;
            inputData[i * 3 + 2] = pixels[i].b;
        }

        return new Tensor(1, imageSize, imageSize, 3, inputData);
    }

    private void ProcessOutput(Tensor output)
    {
        // YOLOv5 출력 처리 (출력을 1차원 배열로 변환)
        var data = output.ToReadOnlyArray();
        int numDetections = output.shape[1];
        int numValues = output.shape[2];

        for (int i = 0; i < numDetections; i++)
        {
            float confidence = data[i * numValues + 4];

            if (confidence < confidenceThreshold)
                continue;

            // 클래스 확률 및 인덱스 계산
            float maxClassProb = 0f;
            int maxClassIndex = 0;

            for (int c = 0; c < m_Labels.Length; c++)
            {
                float classProb = data[i * numValues + 5 + c];
                if (classProb > maxClassProb)
                {
                    maxClassProb = classProb;
                    maxClassIndex = c;
                }
            }

            float x = data[i * numValues + 0];
            float y = data[i * numValues + 1];
            float w = data[i * numValues + 2];
            float h = data[i * numValues + 3];

            // 탐지 결과 출력
            Debug.Log($"검출: {m_Labels[maxClassIndex]}, 신뢰도: {confidence * maxClassProb}, 위치: {x},{y},{w},{h}");

            // 여기서 탐지 결과를 시각화하는 코드를 추가할 수 있습니다.
        }
    }

    private void OnDestroy()
    {
        // ARCameraManager의 이벤트 구독 해제
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived -= OnCameraFrameReceived;
        }

        // 리소스 정리
        m_Worker?.Dispose();
    }
}
