from django import template
import json
register = template.Library()


@register.filter(name="fonk")
def fonksiyon(value):
    y = json.dumps(value)

    return y





