import cv2
import shutil
from PIL import Image, ImageDraw
from collections import Counter
import pickle
import face_recognition
from pathlib import Path
from django.http import HttpResponse
from django.template import loader
from django.contrib import messages
from django.contrib.auth import get_user_model
from .models import Student
from django.contrib.auth import authenticate, login
from django.contrib.auth import login

from django.shortcuts import render, redirect, reverse
from django.core.files.uploadedfile import TemporaryUploadedFile
from members import views
from django.contrib.auth.models import User, Group

from urllib.request import urlopen
from django.core.files import File
from django.core.files.temp import NamedTemporaryFile
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import *


from django.core.files.base import ContentFile
from django.contrib.auth import authenticate, login as auth_login
import base64

import sqlite3


from .models import Student, Attendance
import datetime
import face_recognition
import os
from datetime import datetime

def members(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render())


def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


def first(request):
    template = loader.get_template('first.html')
    return HttpResponse(template.render())


"""
def login(request):
  template = loader.get_template('login.html')
  return HttpResponse(template.render())
"""


#def register(request):
  #  template = loader.get_template('register.html')
   # return HttpResponse(template.render())


def adminstrator(request):
    tid = request.GET.get('tid', None)
    if tid != None:
        try:
            user = User.objects.get(id=tid)
            user.delete()
            messages.success(request, "Teacher Delete Successfully")
            # Handle successful deletion (e.g., show a success message or redirect)
        except User.DoesNotExist:
            messages.error(request, "Something went wrong")
            pass
        return redirect('/adminstrator')
    else:
      teacher_group = Group.objects.get(name='teacher')
      # Get all users in the 'teacher' group
      teachersx = User.objects.filter(groups__in=[teacher_group])
      # Pass the context data directly to the render() function
      return render(request, 'adminstrator.html', {'teachers': teachersx})


def student(request):
    template = loader.get_template('student.html')
    return HttpResponse(template.render())


def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render())


def about(request):
    template = loader.get_template('about.html')
    return HttpResponse(template.render())


def dashboard(request):
    template = loader.get_template('dashboard.html')
    return HttpResponse(template.render())


def course(request):
    template = loader.get_template('course.html')
    return HttpResponse(template.render())


def contact(request):
    template = loader.get_template('contact.html')
    return HttpResponse(template.render())


def blog(request):
    template = loader.get_template('blog.html')
    return HttpResponse(template.render())


def blog_2(request):
    template = loader.get_template('blog-2.html')
    return HttpResponse(template.render())

def camera(request):
    template = loader.get_template('camera.html')
    return HttpResponse(template.render())


def adRegister(request):
    # Your view logic goes here
    return render(request, 'adRegister.html')


def event(request):
    template = loader.get_template('event.html')
    return HttpResponse(template.render())

def teacher(request):
    template = loader.get_template('teacher.html')
    return HttpResponse(template.render())


def stdashboard(request):
    template = loader.get_template('stdashboard.html')
    return HttpResponse(template.render())

    


def get_teachers(request):
    # Get the 'teacher' group (replace 'Teacher' with your desired group name)
    teacher_group = Group.objects.get(name='Teacher')

    # Get all users in the 'teacher' group
    teachers = User.objects.filter(groups__in=[teacher_group])

    # Now 'teachers' contains all the users in the 'teacher' group
    # You can use this queryset for further processing or pass it to the template

    return render(request, 'teachers.html', {'teachers': teachers})


# def adLogin(request):
#     if request.user.is_authenticated:
#         redirect('members/adminstrator')
#     else:
#         return render(request, "adLogin.html")


def teRegister(request):
    template = loader.get_template('teRegister.html')
    return HttpResponse(template.render())


def teLogin(request):
    template = loader.get_template('teLogin.html')
    return HttpResponse(template.render())


def markAttendance(request):
    template = loader.get_template('markAttendance.html')
    return HttpResponse(template.render())

def login_view(request):
    if request.method == 'POST':
        # handle login logic here
        pass
    return render(request, 'login.html')

def dashboard(request):
    # Your view logic here
    return render(request, 'dashboard.html')

from django.contrib.auth import logout
from django.shortcuts import redirect

def user_logout(request):
    logout(request)
    return redirect('teLogin.html')  # Redirect to login page or any other page

from django.contrib.auth import authenticate, login as auth_login
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .models import Student

def register(request):
    if request.method == "POST":
        username = request.POST['email']
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        phone = request.POST['phone']
        department = request.POST['department']
        semester = request.POST.get('semester', 1)
        gender = request.POST['gender']
        image = request.FILES.get('image')  # Use request.FILES for file upload
        roll_no = request.POST['roll_no']

        # Check if passwords match
        if password1 == password2:
            # Check if a user with the given username (email) already exists
            if User.objects.filter(username=username).exists():
                return render(request, 'register.html', {'error': 'User already exists'})
            else:
                # Create the new user
                user = User.objects.create_user(username=username, password=password1, first_name=first_name, last_name=last_name)
                student = Student.objects.create(user=user, phone=phone, department=department, semester=semester, image=image, gender=gender, roll_no=roll_no)

                # Redirect to the login page or any other page after successful registration
                return render(request, 'login.html')
        else:
            return render(request, 'register.html', {'error': 'Passwords do not match'})

    return render(request, 'register.html')

def user_login(request):
    if request.user.is_authenticated:
        return render(request, 'stdashboard.html')

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            return render(request, 'stdashboard.html')

    return render(request, 'login.html')

def logout_view(request):
    logout_view = LogoutView.as_view(next_page='members:login') # Redirect to the teacher login page
    return logout_view(request)

from django.shortcuts import render
from .models import Student



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login

def adLogin(request):
    if request.user.is_authenticated:
        if request.user.groups.filter(name='admin').exists():
           return render(request, 'adminstrator.html')  # Replace 'adminstrator' with your admin dashboard URL
        else:
            return render(request, 'dashboard.html')  # Replace 'dashboard' with your regular user dashboard URL

    if request.method == 'POST':
        # Handle login form submission as before
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if user.groups.filter(name='admin').exists():
                return render(request, 'adminstrator.html')
            else:
                return render(request, 'dashboard.html')

    return render(request, 'adLogin.html')
    
from django.contrib.auth import logout as django_logout
from django.shortcuts import redirect

def custom_logout(request):
    user = request.user
    django_logout(request)
    if user.groups.filter(name='admin').exists():
        return redirect('members:adLogin')
    elif user.groups.filter(name='teacher').exists():
        return redirect('members:teLogin')
    else:
        return redirect('members:login')

def teRegister(request):
    if request.method == "POST":
        username = request.POST['email']
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        department = request.POST['department']
        phone = request.POST['phone']
        gender = request.POST['gender']
        # return render(request, 'template.html', {'ref': ref_value})

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            # Note: this should be a named url pattern, not "register.html"
            return redirect('/teRegister')

        user = User.objects.create_user(
            first_name=first_name, last_name=last_name, username=username, password=password1)
        teachers = Teacher.objects.create(
            user=user, department=department, phone=phone, gender=gender, type="admin")

        # Get or create the 'teacher' group (replace 'Teacher' with your desired group name)
        teacher_group, _ = Group.objects.get_or_create(name='teacher')

        # Add the user to the 'teacher' group
        user.groups.add(teacher_group)
        user.save()
        teachers.save()

        messages.success(
            request, "Your account has been successfully created.")
        return render(request, 'adminstrator.html')
    ref_value = request.GET.get('ref', None)
    return render(request, "teLogin.html", {'ref': ref_value})



from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.models import Group

def teLogin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            if user.groups.filter(name='teacher').exists():
                login(request, user)
                return redirect('/teacher_profile') # Redirect to the teacher's profile page
            else:
                return render(request, 'teLogin.html', {'error_message': 'You are not a teacher.'})
        else:
            return render(request, 'teLogin.html', {'error_message': 'Invalid login credentials.'})
    return render(request, 'teLogin.html')
from django.contrib.auth import authenticate, login, logout

from django.contrib.auth import views as auth_views



def teacher_profile(request):
    if request.user.is_authenticated:
        try:
            teacher = Teacher.objects.get(user=request.user)
            print("Teacher's Email:", teacher.user.email)  # Debugging line
        except Teacher.DoesNotExist:
            teacher = None

        return render(request, 'teacher_profile.html', {'user': request.user, 'teacher': teacher})
    else:
        return redirect('/teLogin')




def some_view(request):
    if request.method == 'GET':
        department = request.GET.get('department', '')
        semester = request.GET.get('semester', '')
        class_name = request.GET.get('class_name', '')
        roll_no = request.GET.get('roll_no', '')

        student = Student.objects.all()

        if department:
            student = student.filter(department__icontains=department)
        if semester:
            student = student.filter(semester=semester)
        if class_name:
            student = student.filter(class_name__icontains=class_name)
        if roll_no:
            student = student.filter(roll_no__icontains=roll_no)

    return render(request, 'some_view.html', {'rows': student})




def filter_students(request):
   # if request.method == 'POST':
    department = request.get('department')
    semester = request.get('semester')
    class_name = request.get('class_name')
    roll_no = request.get('roll_no')
    print(request.GET)

    applicants = Applicant.objects.all()
    logger.info(f'Initially, found {applicants.count()} applicants.')

    if department:
        applicants = applicants.filter(department=department)

    if semester:
        applicants = applicants.filter(
            semester=int(semester))  # convert semester to int

    if class_name:
        applicants = applicants.filter(class_name=class_name)

    if roll_no:
        applicants = applicants.filter(roll_no=roll_no)

    logger.info(f'After filtering, found {applicants.count()} applicants.')

    return render(request, 'filter_students.html', {'applicants': applicants})

   # return render(request, 'filter_students.html')

from django.shortcuts import render
from .models import Student, Attendance, CapturedImage
from django.http import JsonResponse
import json
import os
import base64
from django.core.files.base import ContentFile
from datetime import datetime
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils.timezone import now
import face_recognition

def capture(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        image_data = data.get('image', '')
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]

        image_file = ContentFile(base64.b64decode(imgstr))

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path_relative = default_storage.save(f'db_facerecog/training_video/webcam_image_{timestamp}.png', image_file)

        file_path_absolute = os.path.join(settings.MEDIA_ROOT, file_path_relative)
        if not os.path.exists(file_path_absolute):
            return JsonResponse({'success': False, 'message': 'File not found: ' + file_path_absolute})

        faces_detected = recognize(file_path_absolute)

        if not faces_detected:
            os.remove(file_path_absolute)
            return JsonResponse({'success': False, 'message': 'No Face found'})

        # Associate the CapturedImage with the Student (assuming request.user represents the logged-in student)
        try:
            student = Student.objects.get(user=request.user)
        except Student.DoesNotExist:
            print(f"Student not found for user: {request.user.username}")
            return JsonResponse({'success': False, 'message': 'Student not found'})

        # Create the CapturedImage and associate it with the student
        captured_image = CapturedImage(image=file_path_relative)
        captured_image.student = student
        captured_image.save()

        # Get the current time as a string in HH:MM:SS format
        current_time = datetime.now().strftime('%H:%M:%S')

        # Create the Attendance entry
        attendance = Attendance(student=student, date=datetime.now().date(), time=current_time)
        attendance.save()

        return JsonResponse({'success': True, 'message': 'You are present'})

    return JsonResponse({'success': False})

from PIL import Image

def recognize(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return []

    try:
        with Image.open(image_path) as img:
            image_to_check = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image_to_check)
            return face_locations
    except Exception as e:
        print("Error loading image:", e)
        return []


from django.utils import timezone

def calculate_attendance_percentage(student):
    today = timezone.now().date()  # Use the timezone module here
    thirty_days_ago = today - timedelta(days=30)

    total_attended_days = Attendance.objects.filter(
        student=student,
        date__gte=thirty_days_ago,
        date__lte=today,
    ).count()

    total_days = 30
    attendance_percentage = (total_attended_days / total_days) * 100 if total_days > 0 else 0

    return round(attendance_percentage, 2)

def view_attendance_percentage(request):
    all_attendances = {}
    image_dir = 'db_facerecog/training_video/'

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            faces_detected = recognize(image_path)
            if faces_detected:
                student_name = filename.split('_')[0]
                if student_name not in all_attendances:
                    all_attendances[student_name] = {'total_days': 0, 'present_days': 0}
                all_attendances[student_name]['total_days'] += 1
                all_attendances[student_name]['present_days'] += 1

    for name, attendance_data in all_attendances.items():
        total_days = attendance_data['total_days']
        present_days = attendance_data['present_days']
        attendance_percentage = (present_days / total_days) * 100 if total_days > 0 else 0
        attendance_data['percentage'] = round(attendance_percentage, 2)

    return render(request, 'attendance_percentage.html', {'attendances': all_attendances})

def recognize_faces_image(request):
    return render(request, 'recognize.html')


from django.shortcuts import render
from .models import Attendance

from django.db.models import Prefetch

from django.db.models import OuterRef, Subquery


def view_attendance(request):
    # Fetch students who captured images
    students_with_images = Student.objects.filter(image__isnull=False).select_related('user')

    # Fetch latest attendance records for each student
    attendances = Attendance.objects.filter(
        student=OuterRef('pk')
    ).order_by('-date', '-time')

    latest_attendances = Student.objects.annotate(
        latest_attendance_date=Subquery(attendances.values('date')[:1]),
        latest_attendance_time=Subquery(attendances.values('time')[:1]),
    ).filter(id__in=[student.id for student in students_with_images])

    # Get the IDs of students who have no attendance records
    students_without_attendance = Student.objects.exclude(id__in=latest_attendances.values_list('id', flat=True))

    # Create a list of dictionaries for students without attendance records
    for student in students_without_attendance:
        latest_attendances = latest_attendances.union(
            Student.objects.filter(id=student.id).annotate(
                latest_attendance_date=None,
                latest_attendance_time=None
            )
        )

    # Pass the latest attendance data to the template
    return render(request, 'attendance.html', {'attendances': latest_attendances})


from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Student, Attendance, CapturedImage

@login_required
def stdashboard(request):
    try:
        # Fetching the student object using the current logged-in user
        student = Student.objects.get(user=request.user)
    except Student.DoesNotExist:
        return render(request, 'stdashboard.html', {'message': 'You are not associated with a student record.'})

    context = {
        'student': student,
    }

    return render(request, 'stdashboard.html', context)

@login_required
def view_my_attendance(request):
    try:
        student = Student.objects.get(user=request.user)
    except Student.DoesNotExist:
        return render(request, 'stdashboard.html', {'message': 'You are not associated with a student record.'})

    my_attendances = Attendance.objects.filter(student=student).order_by('-date', '-time')
    
    current_date = timezone.localdate() # Get current date
    all_captured_images = CapturedImage.objects.filter(
        student=student,
        timestamp__date=current_date
    ).order_by('-timestamp')
    
    # Keep track of IDs for which images have already been added
    added_ids = set()
    captured_images = []

    for image in all_captured_images:
        if image.student.id not in added_ids:
            added_ids.add(image.student.id)
            captured_images.append(image) # Add the first image for each unique student ID

    return render(request, 'my_attendance.html', {'attendances': my_attendances, 'captured_images': captured_images})

def view_images(request):
    date_query = request.GET.get('date', None)
    if date_query:
        date = datetime.strptime(date_query, '%Y-%m-%d')
        start_date = timezone.make_aware(date.replace(hour=0, minute=0, second=0, microsecond=0))
        end_date = timezone.make_aware(date.replace(hour=23, minute=59, second=59, microsecond=999999))
    else:
        today = timezone.now().date()  # Get current date
        start_date = timezone.make_aware(datetime.combine(today, datetime.min.time()))
        end_date = timezone.make_aware(datetime.combine(today, datetime.max.time()))

    images = CapturedImage.objects.filter(timestamp__range=(start_date, end_date)).order_by('-timestamp')
    for image in images:
        image.timestamp = image.timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format date for display
    return render(request, 'view_images.html', {'images': images, 'date_query': date_query})


