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
    public Text debugTextPrefab;
    
    public float maxBoundingBoxDistance = 5.0f;
    
    private Model runtimeModel;
    private IWorker worker;
    private const int IMG_WIDTH = 224;
    private const int IMG_HEIGHT = 224;
    private GameObject currentBboxObject;
    private Text currentDebugText;
    private LineRenderer boxLineRenderer;
    
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

        InitializeBoundingBox();

        runtimeModel = ModelLoader.Load(onnxModelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
        Debug.Log("ONNX 모델 로드 완료");
    }

    private void InitializeBoundingBox()
    {
        currentBboxObject = new GameObject("BoundingBox");
        boxLineRenderer = currentBboxObject.AddComponent<LineRenderer>();

        // LineRenderer 설정
        boxLineRenderer.startWidth = 0.02f;
        boxLineRenderer.endWidth = 0.02f;
        boxLineRenderer.positionCount = 5;
        boxLineRenderer.loop = true;
        boxLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        boxLineRenderer.startColor = Color.green;
        boxLineRenderer.endColor = Color.green;
        boxLineRenderer.useWorldSpace = true;
        
        currentBboxObject.SetActive(false);
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

        Texture2D resized = ScaleTexture(texture, IMG_WIDTH, IMG_HEIGHT);
        Debug.Log("텍스처 리사이즈 완료");

        if (debugRawImage != null)
            debugRawImage.texture = resized;

        using (Tensor input = new Tensor(resized, channels: 3))
        {
            try
            {
                worker.Execute(input);
        
                Tensor bboxOutput = worker.PeekOutput("output_0");
                Tensor classOutput = worker.PeekOutput("output_1");

                float x = bboxOutput[0];
                float y = bboxOutput[1];
                float w = bboxOutput[2];
                float h = bboxOutput[3];
        
                string classProbabilities = "";
                for (int i = 0; i < classOutput.length; i++)
                {
                    classProbabilities += $"[{i}]={classOutput[i]:F3} ";
                }
                Debug.Log("클래스 확률 분포: " + classProbabilities);

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
        
                float detectionThreshold = 0.9f;
                if (maxProb < detectionThreshold)
                {
                    if (currentBboxObject != null)
                    {
                        currentBboxObject.SetActive(false);
                    }
                    if (currentDebugText != null)
                    {
                        currentDebugText.gameObject.SetActive(false);
                    }
                    Debug.Log("객체 감지 실패 (신뢰도 낮음)");
                }
                else
                {
                    string detectionText = $"감지된 쓰레기: {className} ({maxProb:P1})";
                    Debug.Log($"모델 추론 완료: bbox=({x:F2}, {y:F2}, {w:F2}, {h:F2}), class: {className} ({maxProb:P1})");
                    
                    DisplayImprovedBoundingBox(x, y, w, h, detectionText);
                }

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
    
    private void DisplayImprovedBoundingBox(float x, float y, float width, float height, string text)
    {
        Camera arCamera = arCameraManager.GetComponent<Camera>();
    
        float depth = maxBoundingBoxDistance;
    
        float halfWidth = width / 2.0f;
        float halfHeight = height / 2.0f;
    
        Vector2[] corners = new Vector2[4];
        corners[0] = new Vector2(x - halfWidth, y - halfHeight); // 좌하단
        corners[1] = new Vector2(x + halfWidth, y - halfHeight); // 우하단
        corners[2] = new Vector2(x + halfWidth, y + halfHeight); // 우상단
        corners[3] = new Vector2(x - halfWidth, y + halfHeight); // 좌상단
    
        Vector3[] worldCorners = new Vector3[4];
        for (int i = 0; i < 4; i++)
        {
            Vector3 screenPos = new Vector3(
                corners[i].x * Screen.width,
                corners[i].y * Screen.height,
                depth
            );
            worldCorners[i] = arCamera.ScreenToWorldPoint(screenPos);
        }
    
        if (boxLineRenderer != null)
        {
            boxLineRenderer.SetPosition(0, worldCorners[0]);
            boxLineRenderer.SetPosition(1, worldCorners[1]);
            boxLineRenderer.SetPosition(2, worldCorners[2]);
            boxLineRenderer.SetPosition(3, worldCorners[3]);
            boxLineRenderer.SetPosition(4, worldCorners[0]);
        
            currentBboxObject.transform.position = Vector3.zero;
            currentBboxObject.SetActive(true);
        }
    
        if (currentDebugText == null && debugTextPrefab != null)
        {
            GameObject textObj = Instantiate(debugTextPrefab.gameObject);
            currentDebugText = textObj.GetComponent<Text>();
        
            currentDebugText.color = Color.white;
            currentDebugText.fontStyle = FontStyle.Bold;
            currentDebugText.alignment = TextAnchor.MiddleCenter;
        
            Shadow shadow = currentDebugText.gameObject.GetComponent<Shadow>();
            if (shadow == null)
            {
                shadow = currentDebugText.gameObject.AddComponent<Shadow>();
            }
            shadow.effectColor = Color.black;
            shadow.effectDistance = new Vector2(2f, -2f);
        }
    
        if (currentDebugText != null)
        {
            Vector3 centerPos = (worldCorners[0] + worldCorners[2]) / 2f;
        
            float boxHeight = Vector3.Distance(worldCorners[0], worldCorners[3]);
            Vector3 textPos = centerPos + arCamera.transform.up * boxHeight * 0.5f;
            currentDebugText.transform.position = textPos;
        
            currentDebugText.transform.rotation = arCamera.transform.rotation;
        
            currentDebugText.text = text;
        
            float distanceToCamera = Vector3.Distance(centerPos, arCamera.transform.position);
            float textScaleFactor = distanceToCamera * 0.05f;
            currentDebugText.transform.localScale = new Vector3(textScaleFactor, textScaleFactor, textScaleFactor);
        
            if (currentDebugText.canvas != null)
            {
                currentDebugText.canvas.sortingOrder = 999;
            }
        
            currentDebugText.gameObject.SetActive(true);
        
            Debug.Log($"텍스트 설정: 위치={textPos}, 스케일={textScaleFactor}, 거리={distanceToCamera}");
        }
        else
       {
            Debug.LogError("디버그 텍스트를 생성하지 못했습니다. debugTextPrefab이 할당되었는지 확인하세요.");
        }
    }


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