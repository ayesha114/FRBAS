from django.urls import path
from members import views
from django.contrib.auth import views as auth_views
from .views import view_images
from members.views import login_view
app_name = 'members'

urlpatterns = [
    path('', views.members, name='members'),
    path('home/', views.home, name='home'),
    path('first/', views.first, name='first'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('recognize/', views.recognize_faces_image, name='recognize'),
    path('attendance_percentage/', views.view_attendance_percentage, name='attendance_percentage'),
    path('adLogin/', views.adLogin, name='adLogin'),
    path('adRegister/', views.adRegister, name='adRegister'),
    path('teRegister/', views.teRegister, name='teRegister'),
    path('teLogin/', views.teLogin, name='teLogin'),
    path('logout/', views.custom_logout, name='logout'),
    path('teacher_profile/', views.teacher_profile, name='teacher_profile'),
    path('adminstrator/', views.adminstrator, name='adminstrator'),
    path('student/', views.student, name='student'),
    path('index/', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('course/', views.course, name='course'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('contact/', views.contact, name='contact'),
    path('blog/', views.blog, name='blog'),
    path('blog_2/', views.blog_2, name='blog_2'),
    path('event/', views.event, name='event'),
    path('markAttendance/', views.markAttendance, name='markAttendance'),
    path('camera/', views.camera, name='camera'),
    path('some_view/', views.some_view, name='some_view'),
    path('capture/', views.capture, name='capture'),
    path('view_images/', views.view_images, name='view_images'),
    path('view_attendance/', views.view_attendance, name='view_attendance'),
    path('stdashboard/', views.stdashboard, name='stdashboard'),
    path('my_attendance/', views.view_my_attendance, name='view_my_attendance'),
    path('teacher/', views.teacher, name='teacher'),

]

