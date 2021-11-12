from . import goods_bp

@goods_bp.route('/goods')
def get_goods():
    return 'get goods'