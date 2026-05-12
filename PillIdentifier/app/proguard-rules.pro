# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /opt/android-sdk/tools/proguard/proguard-android.txt

# Keep OpenCV classes
-keep class org.opencv.** { *; }

# Keep Gson model classes
-keep class com.pillid.model.** { *; }

# Keep annotations
-keepattributes *Annotation*
