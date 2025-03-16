using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Barracuda;
using UnityEngine.UI;
using System.Collections.Generic;

public class ARTrashDetector : MonoBehaviour
{
    public ARCameraManager arCameraManager;
    public NNModel onnxModelAsset;
    public RawImage debugRawImage;
    public Text debugText;
    public GameObject bboxPrefab;
    
    private Model runtimeModel;
    private IWorker worker;
    private const int IMG_WIDTH = 224;
    private const int IMG_HEIGHT = 224;
    private GameObject currentBboxObject;
    
    private readonly string[] trashCategories = {
        "Aluminium foil", "Bottle cap", "Bottle", "Broken glass", "Can", 
        "Carton", "Cigarette", "Cup", "Lid", "Other litter", 
        "Other plastic", "Paper", "Plastic bag - wrapper", "Plastic container", 
        "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter"
    };

    void Start()
    {
        if (arCameraManager == null)
            Debug.LogError("[ARTrashDetector] arCameraManager가 null입니다!");
        else
            Debug.Log("[ARTrashDetector] arCameraManager 할당됨: " + arCameraManager.name);

        runtimeModel = ModelLoader.Load(onnxModelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
        Debug.Log("ONNX 모델 로드 완료");
    }

    void OnEnable()
    {
        if (arCameraManager == null)
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
            cpuImage.Dispose();
        }

        // 모델 입력 사이즈에 맞게 텍스처 리사이즈
        Texture2D resized = ScaleTexture(texture, IMG_WIDTH, IMG_HEIGHT);
        Debug.Log("텍스처 리사이즈 완료");

        // 디버그용 RawImage 업데이트
        if (debugRawImage != null)
            debugRawImage.texture = resized;

        // 모델 추론 (다중 출력: 바운딩 박스 + 클래스)
        using (Tensor input = new Tensor(resized, channels: 3))
        {
            try
            {
                worker.Execute(input);
        
                // 다중 출력 텐서 가져오기
                Tensor bboxOutput = worker.PeekOutput("output_0");
                Tensor classOutput = worker.PeekOutput("output_1");

                // 바운딩 박스 좌표 (출력값은 0~1 사이)
                float x = bboxOutput[0];
                float y = bboxOutput[1];
                float w = bboxOutput[2];
                float h = bboxOutput[3];
        
                // 클래스 예측 결과: softmax 확률 벡터의 모든 값을 출력해서 디버깅
                string classProbabilities = "";
                for (int i = 0; i < classOutput.length; i++)
                {
                    classProbabilities += $"[{i}]={classOutput[i]:F3} ";
                }
                Debug.Log("클래스 확률 분포: " + classProbabilities);

                // softmax 확률 벡터에서 argmax 계산
                int classIndex = 0;
                float maxProb = 0f;
                for (int i = 0; i < classOutput.length; i++)
                {
                    if (classOutput[i] > maxProb)
                    {
                        maxProb = classOutput[i];
                        classIndex = i;
                    }
                }
        
                string className = trashCategories[classIndex];
        
                // 디버그 텍스트 업데이트 (바운딩 박스 좌표 및 감지 결과)
                if (debugText != null)
                {
                    debugText.text = $"위치: x:{x:F2}, y:{y:F2}, w:{w:F2}, h:{h:F2}\n" +
                                     $"감지된 쓰레기: {className} ({maxProb:P1})";
                }
        
                Debug.Log($"모델 추론 완료: bbox=({x}, {y}, {w}, {h}), class: {className} ({maxProb:P1})");
        
                // AR 환경에 바운딩 박스 표시
                DisplayBoundingBox(x, y, w, h);
        
                bboxOutput.Dispose();
                classOutput.Dispose();
            }
            catch (System.Exception e)
            {
                Debug.LogError("모델 추론 중 오류 발생: " + e.Message);
            }
        }

        Destroy(texture);
        Destroy(resized);
    }
    
    // AR 환경에 바운딩 박스 표시
    private void DisplayBoundingBox(float x, float y, float width, float height)
    {
        // 기존 바운딩 박스 제거
        if (currentBboxObject != null)
            Destroy(currentBboxObject);
            
        if (bboxPrefab != null)
        {
            // 화면 좌표를 AR 환경 좌표로 변환
            Vector2 screenPoint = new Vector2(
                x * Screen.width,
                y * Screen.height
            );
            
            // AR 레이캐스트로 실제 위치 찾기
            var arRaycastManager = FindObjectOfType<ARRaycastManager>();
            List<ARRaycastHit> hits = new List<ARRaycastHit>();
            
            if (arRaycastManager.Raycast(screenPoint, hits, TrackableType.PlaneWithinPolygon))
            {
                var hitPose = hits[0].pose;
                
                // 바운딩 박스 생성
                currentBboxObject = Instantiate(bboxPrefab, hitPose.position, hitPose.rotation);
                
                // 크기 조정
                float scaleFactor = 0.5f;
                currentBboxObject.transform.localScale = new Vector3(
                    width * scaleFactor,
                    height * scaleFactor / 2,
                    0.01f // 두께
                );
            }
        }
    }

    // 텍스처 리사이즈 함수
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
