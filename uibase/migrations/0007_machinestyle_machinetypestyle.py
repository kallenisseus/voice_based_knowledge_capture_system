from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("uibase", "0006_audioupload_subcategory_paths"),
    ]

    operations = [
        migrations.CreateModel(
            name="MachineStyle",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("machine_name", models.CharField(max_length=255, unique=True)),
                ("color_hex", models.CharField(default="#3A78F2", max_length=7)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["machine_name"],
            },
        ),
        migrations.CreateModel(
            name="MachineTypeStyle",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("machine_type", models.CharField(max_length=255, unique=True)),
                ("color_hex", models.CharField(default="#3A78F2", max_length=7)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["machine_type"],
            },
        ),
    ]
