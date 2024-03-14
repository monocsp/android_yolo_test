package com.programminghut.realtime_object

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
//import com.programminghut.realtime_object.ml.BestFloat32
import com.programminghut.realtime_object.ml.FingerBestInt8
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
//    lateinit var model:SsdMobilenetV11Metadata1
//    lateinit var best32:BestFloat32
    lateinit var fingerBestFloat32: FingerBestInt8

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR)).build()
//        model = SsdMobilenetV11Metadata1.newInstance(this)
//        best32 = BestFloat32.newInstance(this)
        fingerBestFloat32 = FingerBestInt8.newInstance(this)

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)

        textureView = findViewById(R.id.textureView)


        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply{
            if(compatList.isDelegateSupportedOnThisDevice){
                // if the device has a supported GPU, add the GPU delegate
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                // if the GPU is not supported, run on 4 threads
                this.setNumThreads(4)
            }
        }

        val tflite = getTfliteInterpreter("finger_best_float32.tflite", this, options);

        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }




            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {

                tflite.run {  }







//                bitmap = textureView.bitmap!!
//                var tensorImage =  TensorImage(DataType.FLOAT32)
//                tensorImage.load(bitmap)
////                var image = TensorImage.fromBitmap(bitmap)
//
//                Log.d("OUTPUTS", "${tensorImage.dataType}")
//
//                tensorImage = imageProcessor.process(tensorImage)
//
////                val outputs = model.process(image)
//
//                val outputs = fingerBestFloat32.process(tensorImage)
//                outputs.classesAsCategoryList;
//                var category =  outputs.outputAsCategoryList;

//                println("outputs data : ${outputs.outputAsCategoryList}")

//                val locations = outputs.locationsAsTensorBuffer.floatArray
//                val classes = outputs.classesAsTensorBuffer.floatArray
//                val scores = outputs.scoresAsTensorBuffer.floatArray
//                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

//                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
//                val canvas = Canvas(mutable)
//
//                val h = mutable.height
//                val w = mutable.width
//                paint.textSize = h/15f
//                paint.strokeWidth = h/85f
//                var x = 0
//                scores.forEachIndexed { index, fl ->
//
//                    x = index
//                    x *= 4
//                    if(fl > 0.5){ //카메라 기능 만들면 끝
//                        paint.setColor(colors.get(index))
//                        paint.style = Paint.Style.STROKE
//                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
//                        paint.style = Paint.Style.FILL
//                        canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
//                    }
//                }

//                imageView.setImageBitmap(mutable)


            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }

    override fun onDestroy() {
        super.onDestroy()
//        model.close()
//        best32.close()
        fingerBestFloat32.close();
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }

    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }

    // 모델 파일 인터프리터를 생성하는 공통 함수
// loadModelFile 함수에 예외가 포함되어 있기 때문에 반드시 try, catch 블록이 필요하다.
    private fun getTfliteInterpreter(modelPath: String, context: Context, options: Interpreter.Options): Interpreter? {
        try {
            return Interpreter(loadModelFile(modelPath, context),options )
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    // 모델을 읽어오는 함수로, 텐서플로 라이트 홈페이지에 있다.
// MappedByteBuffer 바이트 버퍼를 Interpreter 객체에 전달하면 모델 해석을 할 수 있다.
    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String, context: Context): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.startOffset
        val declaredLength: Long = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }



}