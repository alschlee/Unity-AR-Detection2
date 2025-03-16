using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Barracuda;
using UnityEngine.UI;

public class ARTrashDetector : MonoBehaviour
{
    public ARCameraManager arCameraManager; // AR 카메라 매니저를 Inspector에 할당
    public NNModel onnxModelAsset;          // ONNX 모델 할당
    public RawImage debugRawImage;          // (옵션) 디버그용 화면 출력
    public Text debugText;                  // (옵션) 결과 출력용 텍스트

    private Model runtimeModel;
    private IWorker worker;
    private const int IMG_WIDTH = 224;
    private const int IMG_HEIGHT = 224;

    void Start()
    {
        if (arCameraManager == null)
        {
            arCameraManager = FindObjectOfType<ARCameraManager>();
            Debug.LogError("ARCameraManager가 할당되지 않았습니다");
            return;
        }

        runtimeModel = ModelLoader.Load(onnxModelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
        Debug.Log("ONNX 모델 로드 완료");
    }

    void OnEnable()
    {
        if (arCameraManager != null)
        {
            Debug.LogError("ARCameraManager가 할당되지 않았습니다");
            return;
        }

        arCameraManager.frameReceived += OnCameraFrameReceived;
        Debug.Log("AR 카메라 프레임 수신 이벤트 등록 완료");
    }

    void OnDisable()
    {
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived -= OnCameraFrameReceived;
            Debug.Log("AR 카메라 프레임 수신 이벤트 해제 완료");
        }
        worker?.Dispose();
        Debug.Log("워커 리소스 해제");
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        Debug.Log("새로운 AR 프레임 수신");

        XRCpuImage cpuImage;
        if (!arCameraManager.TryAcquireLatestCpuImage(out cpuImage))
        {
            Debug.LogError("AR 프레임을 가져오는 데 실패했습니다.");
            return;
        }
        Debug.Log("CPU 이미지 획득 성공");

        // 변환 옵션 설정 (예: YUV -> RGB 변환)
        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
            outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
            outputFormat = TextureFormat.RGB24,
            transformation = XRCpuImage.Transformation.MirrorY
        };

        // 변환할 Texture2D 생성
        Texture2D texture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGB24, false);
        var rawTextureData = texture.GetRawTextureData<byte>();
        try
        {
            cpuImage.Convert(conversionParams, rawTextureData);
            texture.Apply();
            Debug.Log("CPU 이미지 변환 및 텍스처 적용 완료");
        }
        catch (System.Exception e)
        {
            Debug.LogError("이미지 변환 중 오류 발생: " + e.Message);
            cpuImage.Dispose();
            return;
        }
        finally
        {
            // 항상 ARCpuImage를 Dispose 해야 함
            cpuImage.Dispose();
        }

        // Resize Texture2D to model input size
        Texture2D resized = ScaleTexture(texture, IMG_WIDTH, IMG_HEIGHT);
        Debug.Log("텍스처 리사이즈 완료");

        // (옵션) 디버그 UI 업데이트
        if (debugRawImage != null)
            debugRawImage.texture = resized;

        // 모델 추론
        using (Tensor input = new Tensor(resized, channels: 3))
        {
            try
            {
                worker.Execute(input);
                Tensor output = worker.PeekOutput();

                // 추론 결과 해석 (예: bounding box 좌표)
                float x = output[0];
                float y = output[1];
                float w = output[2];
                float h = output[3];

                if (debugText != null)
                    debugText.text = $"BBox: x:{x:F2}, y:{y:F2}, w:{w:F2}, h:{h:F2}";

                Debug.Log($"모델 추론 완료: x={x}, y={y}, w={w}, h={h}");

                output.Dispose();
            }
            catch (System.Exception e)
            {
                Debug.LogError("모델 추론 중 오류 발생: " + e.Message);
            }
        }

        Destroy(texture);
        Destroy(resized);
    }

    // 간단한 텍스처 리사이즈 함수
    Texture2D ScaleTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        Texture2D result = new Texture2D(targetWidth, targetHeight, source.format, false);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }
}
