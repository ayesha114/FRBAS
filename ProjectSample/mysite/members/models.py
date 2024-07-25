from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.contrib.auth import get_user_model
# Create your models here.
User = get_user_model()
class Student(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)
    phone = models.CharField(max_length=10)
    gender = models.CharField(max_length=10)
    type = models.CharField(max_length=15)
    department = models.CharField(max_length=100, default='No department')
    semester = models.IntegerField()
    roll_no = models.CharField(max_length=100)
    image = models.ImageField(upload_to='db_facerecog/training_video/')
    registration_date = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def present_days(self):
        return self.attendance_set.filter(date__lt=timezone.now().date()).count()

    def __str__(self):
        return self.user.username
        
class Teacher(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=10)
    gender = models.CharField(max_length=10)
    type = models.CharField(max_length=15)
    department = models.CharField(max_length=100, default='No department')
    registration_date = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):
        return self.user.first_name 

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.CharField(max_length=8)
    
    def __str__(self):
        return f"{self.student.first_name} {self.student.last_name} - {self.date} {self.time}"

class PersonAttendance(models.Model):
    name = models.CharField(max_length=100, unique=True)
    total_images = models.PositiveIntegerField(default=0)
    present_images = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.name


class CapturedImage(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='db_facerecog/training_video/')
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return str(self.timestamp)




class Application(models.Model):
   student  = models.ForeignKey(Student, on_delete=models.CASCADE)

