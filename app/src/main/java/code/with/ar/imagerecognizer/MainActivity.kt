package code.with.ar.imagerecognizer

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.FileUtils
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView

import androidx.core.content.ContextCompat
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

import code.with.ar.imagerecognizer.ml.SsdMobilenetV11Metadata1
//import kotlinx.coroutines.flow.internal.NoOpContinuation.context

//import kotlin.coroutines.jvm.internal.CompletedContinuation.context

class MainActivity : AppCompatActivity() {
    val paint=Paint()
    lateinit var labels:List<String>
    lateinit var imageProcessor:ImageProcessor
lateinit var model:SsdMobilenetV11Metadata1
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    val colors= listOf<Int>(
        Color.BLUE, Color.GREEN, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK, Color.DKGRAY, Color.MAGENTA, Color.YELLOW
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()
        labels = FileUtil.loadLabels(this, "mobilenet_objectdetection_labels.txt")
imageProcessor=ImageProcessor.Builder().add(ResizeOp(300,300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model=SsdMobilenetV11Metadata1.newInstance(this)

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {


            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {

            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap = textureView.bitmap!!

//Creates inputs for reference
                var image=TensorImage.fromBitmap(bitmap)
                image=imageProcessor.process(image)





//runs model inference and gets result
                val outputs =model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes= outputs.classesAsTensorBuffer.floatArray
                val scores=outputs.scoresAsTensorBuffer.floatArray
//                val numberOfDetections=outputs.numberOfDetectionsAsTensorBuffer.floatArray

var mutable=bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas=Canvas(mutable)

                val h=mutable.height
                val w=mutable.width
                paint.textSize=h/15f
                paint.strokeWidth=h/85f
                var x=0
                scores.forEachIndexed { index, fl->
                    x=index
                    x*= 4
                    if(fl>0.5){
                        paint.setColor(colors.get(index))
                        paint.style= Paint.Style.STROKE
                        canvas.drawRect(
                            RectF(locations.get(x+1)*w,locations.get(x)*h ,locations.get(x+3)*w,locations.get(x+2)*h)  , paint)
                                paint.style=Paint.Style.FILL
                                canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(),locations.get(x+1)*w , locations.get(x)*h, paint)
                    }
                }
imageView.setImageBitmap(mutable)
////Releases model resources if no longer used
//                model.close()
            }
        }
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }
    override fun onDestroy(){
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    fun openCamera() {
        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    var surfaceTexture = textureView.surfaceTexture
                    var surface = Surface(surfaceTexture)
                    var captureRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequest.addTarget(surface)

                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(session: CameraCaptureSession) {
                                session.setRepeatingRequest(captureRequest.build(), null, null)
                            }

                            override fun onConfigureFailed(session: CameraCaptureSession) {

                            }
                        },
                        handler
                    )

                }

                override fun onDisconnected(camera: CameraDevice) {

                }

                override fun onError(camera: CameraDevice, error: Int) {

                }
            },
            handler
        )

    }

    fun get_permission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)

        }
    }

    //override
    fun onRequestPermissionResult(
        requestCode: Int,
        permission: Array<out String>,
        grantResults: IntArray
    ) {

        super.onRequestPermissionsResult(requestCode, permission, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()
        }
    }
}






